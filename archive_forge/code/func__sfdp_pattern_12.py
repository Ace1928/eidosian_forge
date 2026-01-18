import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_pattern_12(query, key, value, inv_scale_factor, dropout_p):
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return torch.nn.functional.dropout(torch.matmul(q, k.transpose(-2, -1)).div(inv_scale_factor).softmax(dim=-1), p=dropout_p).matmul(v)