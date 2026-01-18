import math
import random
from typing import List, Optional, Sequence, Tuple, Type
import torch
from xformers.ops import fmha
from xformers.ops.fmha.common import AttentionOpBase

    Generates lists of lengths of query blocks and corresponding key blocks.
    The total number of queries will be bs * q_len and the
    total number of keys will be bs * kv_len.
    max_q_minus_k: maximum allowed num_queries - num_keys.
        For "bottom-right" masks it's 0, we need to have more keys than
        queries, otherwise some queries have no keys to attend to.
        For BlockDiagonalCausalMask it's None, there is no constraint
        on num_queries - num_keys.
        For BlockDiagonalCausalLocalAttentionMask it's equal
        to the window size.
    