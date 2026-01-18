import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_pattern_1(query, key, value, inv_scale):
    return torch.matmul(query, key.transpose(-2, -1)).div(inv_scale).softmax(dim=-1).matmul(value)