import copy
import itertools
import warnings
import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
from torch.ao.nn.intrinsic import _FusedModule
from torch.ao.quantization.quantization_mappings import (
from .utils import get_qparam_dict, has_no_children_ignoring_parametrizations
from torch.ao.quantization.stubs import DeQuantStub, QuantWrapper
from torch.ao.quantization.qconfig import (
from torch.nn.utils.parametrize import type_before_parametrizations
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.observer import (   # noqa: F401
def _register_activation_post_process_hook(module, pre_hook=False):
    assert hasattr(module, 'activation_post_process'), 'Expect activation_post_process attribute already attached to the module'
    if pre_hook:
        handle = module.register_forward_pre_hook(_observer_forward_pre_hook, prepend=True)
    else:
        handle = module.register_forward_hook(_observer_forward_hook, prepend=True)