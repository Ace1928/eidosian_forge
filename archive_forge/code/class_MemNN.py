import torch
import torch.nn as nn
from parlai.utils.torch import neginf
from functools import lru_cache
class MemNN(nn.Module):
    """
    Memory Network module.
    """

    def __init__(self, num_features, embedding_size, hops=1, memsize=32, time_features=False, position_encoding=False, dropout=0, padding_idx=0):
        """
        Initialize memnn model.

        See cmdline args in MemnnAgent for description of arguments.
        """
        super().__init__()
        self.hops = hops

        def embedding(use_extra_feats=True):
            return Embed(num_features, embedding_size, position_encoding=position_encoding, padding_idx=padding_idx)
        self.query_lt = embedding()
        self.in_memory_lt = embedding()
        self.out_memory_lt = embedding()
        self.answer_embedder = embedding()
        self.memory_hop = Hop(embedding_size)

    def forward(self, xs, mems, cands=None, pad_mask=None):
        """
        One forward step.

        :param xs:
            (bsz x seqlen) LongTensor queries to the model

        :param mems:
            (bsz x num_mems x seqlen) LongTensor memories

        :param cands:
            (num_cands x seqlen) or (bsz x num_cands x seqlen)
            LongTensor with candidates to rank
        :param pad_mask:
            (bsz x num_mems) optional mask indicating which tokens
            correspond to padding

        :returns:
            scores contains the model's predicted scores.
            if cand_params is None, the candidates are the vocabulary;
            otherwise, these scores are over the candidates provided.
            (bsz x num_cands)
        """
        state = self.query_lt(xs)
        if mems is not None:
            in_memory_embs = self.in_memory_lt(mems).transpose(1, 2)
            out_memory_embs = self.out_memory_lt(mems)
            for _ in range(self.hops):
                state = self.memory_hop(state, in_memory_embs, out_memory_embs, pad_mask)
        if cands is not None:
            cand_embs = self.answer_embedder(cands)
        else:
            cand_embs = self.answer_embedder.weight
        return (state, cand_embs)