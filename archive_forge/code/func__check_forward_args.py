from __future__ import annotations
import math
import warnings
from typing import Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight, gather_params_ctx
from peft.utils.other import transpose
from .config import LoraConfig
def _check_forward_args(self, x, *args, **kwargs):
    """Check if the arguments are compatible with the configs and state of the model"""
    adapter_names = kwargs.get('adapter_names', None)
    if adapter_names is None:
        return
    if len(x) != len(adapter_names):
        msg = f'Length of `adapter_names` should be the same as the number of inputs, but got {len(adapter_names)} and {len(x)} respectively.'
        raise ValueError(msg)
    if self.merged:
        msg = 'Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first.'
        raise ValueError(msg)
    unique_adapters = set(self.active_adapters)
    for adapter_name in unique_adapters:
        if self.use_dora.get(adapter_name, False):
            msg = 'Cannot pass `adapter_names` when DoRA is enabled.'
            raise ValueError(msg)