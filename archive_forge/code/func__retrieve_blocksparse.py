import logging
import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Union
import torch
from xformers import _has_cpp_library, _is_triton_available
from xformers.components.attention.attention_mask import AttentionMask
@lru_cache(maxsize=128)
def _retrieve_blocksparse(num_heads: int, seq_len: int, block_size: int) -> BlockSparseAttention:
    blocks = seq_len // block_size
    layout_fill = torch.ones((num_heads, blocks, blocks), dtype=torch.long)
    return BlockSparseAttention(layout=layout_fill, block_size=block_size, causal=True)