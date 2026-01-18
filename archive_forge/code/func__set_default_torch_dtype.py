import contextlib
from typing import Type
import torch
import torch.nn as nn
from vllm.config import DeviceConfig, ModelConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.weight_utils import (get_quant_config,
@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)