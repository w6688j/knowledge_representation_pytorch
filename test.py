import torch

model = torch.load('model/FB13/l_0.001_es_0_L_1_em_100_nb_100_n_1000_m_1.0_f_1_mo_0.9_s_0_op_1_lo_0_TransE.ckpt')
ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy()

word = 'market'
entitiy2id = {}

f = open('datasets/WN11/new_entity2id.txt', encoding='utf-8')
for i, line in enumerate(f):
    if i > 2:
        line_split = line.strip().split(' ')
        entitiy2id[line_split[0]] = line_split[1]

f.close()

word2id = entitiy2id[word] if word in entitiy2id.keys() else 0
print(ent_embeddings[int(word2id)])
