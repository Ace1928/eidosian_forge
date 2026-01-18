import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class BlockDiagonalCausalFromBottomRightMask(BlockDiagonalMask):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`, except that each block is causal.
    This mask allows for a non-causal prefix
    NOTE: Each block should have `num_keys >= num_queries` otherwise the forward pass is not
    defined (softmax of vector of `-inf` in the attention)

    Queries and keys are each divided into the same number of blocks.
    A query Q in block i cannot attend to a key which is not in block i,
    nor one which nearer the final key in block i than Q is to the
    final query in block i.
    """

    def __post_init__(self) -> None:
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(zip(self.q_seqinfo.intervals(), self.k_seqinfo.intervals())):
            num_queries = q_end - q_start
            num_keys = k_end - k_start
            if num_keys < num_queries:
                raise ValueError(f'Block #{i} has num_keys={num_keys} and num_queries={num_queries}. Expected `num_keys >= num_queries`')

    def _create_block_mask(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        return LowerTriangularFromBottomRightMask().materialize(shape=shape, dtype=dtype, device=device)