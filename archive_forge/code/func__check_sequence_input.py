import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from ..utils import _log_api_usage_once
from . import functional as F
from .functional import _interpolation_modes_from_int, InterpolationMode
def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else ' or '.join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError(f'{name} should be a sequence of length {msg}.')
    if len(x) not in req_sizes:
        raise ValueError(f'{name} should be a sequence of length {msg}.')