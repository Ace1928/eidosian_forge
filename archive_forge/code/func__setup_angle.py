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
def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(f'If {name} is a single number, it must be positive.')
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)
    return [float(d) for d in x]