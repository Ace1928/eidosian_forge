import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_pattern_2(query, key, value, scale_factor):
    return torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1).matmul(value)