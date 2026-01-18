import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_replacement_12(query, key, value, inv_scale_factor, dropout_p):
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=None, dropout_p=dropout_p, is_causal=False, scale=1.0 / inv_scale_factor)