import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from vllm._C import ops
def _yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype: torch.dtype) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func