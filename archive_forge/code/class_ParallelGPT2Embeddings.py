import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from flash_attn.utils.distributed import all_reduce, reduce_scatter
class ParallelGPT2Embeddings(nn.Module):

    def __init__(self, embed_dim, vocab_size, max_position_embeddings, process_group, padding_idx=None, sequence_parallel=True, device=None, dtype=None):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.word_embeddings = VocabParallelEmbedding(vocab_size, embed_dim, padding_idx=padding_idx, process_group=process_group, **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = ColumnParallelEmbedding(max_position_embeddings, embed_dim, process_group=process_group, **factory_kwargs)

    def forward(self, input_ids, position_ids=None, combine_batch_seqlen_dim=False):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        world_size = torch.distributed.get_world_size(self.process_group)
        embeddings = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            if world_size <= 1:
                embeddings = embeddings + position_embeddings
            else:
                partition_dim = self.position_embeddings.embedding_dim
                rank = torch.distributed.get_rank(self.process_group)
                embeddings[..., rank * partition_dim:(rank + 1) * partition_dim] += position_embeddings
        if combine_batch_seqlen_dim:
            embeddings = rearrange(embeddings, 'b s d -> (b s) d')
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return embeddings if world_size <= 1 else reduce_fn(embeddings, self.process_group)