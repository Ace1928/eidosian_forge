import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_replacement_13(query, key, value, dropout_p):
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), dropout_p=dropout_p, scale=1.0).squeeze(0)