import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from flash_attn.utils.distributed import all_reduce, reduce_scatter
class ColumnParallelEmbedding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, *args, process_group=None, **kwargs):
        self.process_group = process_group
        if process_group is not None:
            world_size = torch.distributed.get_world_size(process_group)
            if embedding_dim % world_size != 0:
                raise ValueError(f'embedding_dim ({embedding_dim}) must be divisible by world_size ({world_size})')
        else:
            world_size = 1
        super().__init__(num_embeddings, embedding_dim // world_size, *args, **kwargs)