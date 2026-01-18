import math
import random
from typing import List, Optional, Sequence, Tuple, Type
import torch
from xformers.ops import fmha
from xformers.ops.fmha.common import AttentionOpBase
def _rand_seqlens_padded_k(r: random.Random, bs: int, q_len: int, kv_len: int) -> Tuple[Sequence[int], Sequence[int]]:
    if q_len > kv_len:
        raise ValueError('need more keys than values')
    if q_len == kv_len:
        q_seqlens = k_seqlens = [kv_len] * bs
    else:
        q_seqlens = _rand_maxed_partition(r, q_len * bs, bs, kv_len)
        k_seqlens = [r.randint(i, kv_len) for i in q_seqlens]
    return (q_seqlens, k_seqlens)