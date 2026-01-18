import torch
import torch.nn as nn
from parlai.utils.torch import neginf
from functools import lru_cache
class Hop(nn.Module):
    """
    Memory Network hop outputs attention-weighted sum of memory embeddings.

    0) rotate the query embeddings 1) compute the dot product between the input vector
    and each memory vector 2) compute a softmax over the memory scores 3) compute the
    weighted sum of the memory embeddings using the probabilities 4) add the query
    embedding to the memory output and return the result
    """

    def __init__(self, embedding_size, rotate=True):
        """
        Initialize linear rotation.
        """
        super().__init__()
        if rotate:
            self.rotate = nn.Linear(embedding_size, embedding_size, bias=False)
        else:
            self.rotate = lambda x: x
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query_embs, in_mem_embs, out_mem_embs, pad_mask):
        """
        Compute MemNN Hop step.

        :param query_embs:
            (bsz x esz) embedding of queries

        :param in_mem_embs:
            bsz list of (num_mems x esz) embedding of memories for activation

        :param out_mem_embs:
            bsz list of (num_mems x esz) embedding of memories for outputs

        :param pad_mask
            (bsz x num_mems) optional mask indicating which tokens correspond to
            padding

        :returns:
            (bsz x esz) output state
        """
        attn = torch.bmm(query_embs.unsqueeze(1), in_mem_embs).squeeze(1)
        if pad_mask is not None:
            attn[pad_mask] = neginf(attn.dtype)
        probs = self.softmax(attn)
        memory_output = torch.bmm(probs.unsqueeze(1), out_mem_embs).squeeze(1)
        output = memory_output + self.rotate(query_embs)
        return output