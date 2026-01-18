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
def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return (int(size), int(size))
    if isinstance(size, Sequence) and len(size) == 1:
        return (size[0], size[0])
    if len(size) != 2:
        raise ValueError(error_msg)
    return size