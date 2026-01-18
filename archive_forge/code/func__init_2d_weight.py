from typing import Callable, List
import torch
import torch.distributed
from .differentiable_collectives import (
from .seqpar import sequence_parallel_leading_matmul, sequence_parallel_trailing_matmul
def _init_2d_weight(weight: torch.Tensor, init_method: Callable[[torch.Tensor], torch.Tensor], process_group: torch.distributed.ProcessGroup, partition_dim: int) -> None:
    rank = process_group.rank()
    world_size = process_group.size()
    nrows, ncols = weight.shape
    if partition_dim == 0:
        full_weight = weight.new_empty(nrows * world_size, ncols)
        my_weight_slice = full_weight[rank::world_size, :]
    else:
        full_weight = weight.new_empty(nrows, ncols * world_size)
        my_weight_slice = full_weight[:, rank::world_size]
    init_method(full_weight)
    with torch.no_grad():
        weight.copy_(my_weight_slice)