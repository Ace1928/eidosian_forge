from typing import Callable, Optional
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from .initialize import get_model_parallel_rank, get_model_parallel_world_size
from .mappings import (
from .utils import VocabUtility, divide_and_check_no_remainder
class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None, max_norm: Optional[float]=None, norm_type: float=2.0, scale_grad_by_freq: bool=False, sparse: bool=False, init_method: Callable[[torch.Tensor], torch.Tensor]=init.xavier_normal_, keep_master_weight_for_test: bool=False) -> None:
        super(ParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = scale_grad_by_freq
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None
        world_size = get_model_parallel_world_size()
        self.embedding_dim_per_partition = divide_and_check_no_remainder(self.embedding_dim, world_size)
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim_per_partition))
        _initialize_affine_weight(self.weight, self.num_embeddings, self.embedding_dim, self.embedding_dim_per_partition, 1, init_method, stride=1, return_master_weight=False)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = F.embedding(input_parallel, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        output = gather_from_model_parallel_region(output_parallel)
        return output