import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_pattern_3(query, key, value, inv_scale_factor, dropout_p):
    return torch.nn.functional.dropout(torch.matmul(query, key.transpose(-2, -1)).div(inv_scale_factor).softmax(dim=-1), p=dropout_p).matmul(value)