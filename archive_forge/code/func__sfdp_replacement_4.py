import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_replacement_4(query, key, value, scale_factor, dropout_p):
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=None, dropout_p=dropout_p, is_causal=False, scale=scale_factor)