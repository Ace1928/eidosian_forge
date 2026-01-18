import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def _is_fp16_bf16_tensor(tensor):
    return (is_torch_tensor(tensor) or hasattr(tensor, 'dtype')) and tensor.dtype in (torch.float16, torch.bfloat16)