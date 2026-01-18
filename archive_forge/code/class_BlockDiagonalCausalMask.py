import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class BlockDiagonalCausalMask(BlockDiagonalMask):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`, except that each block is causal.

    Queries and Keys are each divided into the same number of blocks.
    A query Q in block i cannot attend to a key which is not in block i,
    nor one which is farther from the initial key in block i than Q
    is from the initial query in block i.
    """

    def _create_block_mask(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        return LowerTriangularMask().materialize(shape, dtype=dtype, device=device)