import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_replacement_11(query, key, value, inv_scale):
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=None, dropout_p=0.0, is_causal=False, scale=1.0 / inv_scale)