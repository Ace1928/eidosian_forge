import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_pattern_6(query, key, value, attn_mask, dropout_p):
    attn_weight = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    return attn_weight @ value