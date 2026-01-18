import logging
import math
from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.init import (
def _init_weights_small(module: nn.Module, name: str='', head_bias: float=0.0, gain: float=1.0, deepnorm_style: bool=False, **kwargs):
    """Follow the `Transformer Without Tears`_ initialization for self-attention"""
    if is_ffn(name):
        _maybe_init_tensor(module, 'weight', torch.nn.init.xavier_uniform_, gain=gain)
        _maybe_init_tensor(module, 'bias', nn.init.normal_, std=1e-06)
    elif is_mha_input_projection(name) or isinstance(module, nn.Linear):
        if deepnorm_style and ('q_proj' in name.split('.') or 'k_proj' in name.split('.')):
            gain = 1.0
        _maybe_init_tensor(module, 'weight', _small_init_, gain=gain)
        _maybe_init_tensor(module, 'bias', nn.init.zeros_)
    elif isinstance(module, nn.Conv2d):
        _maybe_init_tensor(module, 'weight', _lecun_normal)
        _maybe_init_tensor(module, 'bias', nn.init.zeros_)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
    else:
        _maybe_report_no_init(module, name)
    if not hasattr(module, 'init_weights'):
        for child_name, child_module in module.named_children():
            _init_weights_small(child_module, f'{name}.{child_name}', head_bias, gain)