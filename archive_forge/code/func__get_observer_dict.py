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
def _get_observer_dict(mod, target_dict, prefix=''):
    """Traverse the modules and save all observers into dict.
    This is mainly used for quantization accuracy debug
    Args:
        mod: the top module we want to save all observers
        prefix: the prefix for the current module
        target_dict: the dictionary used to save all the observers
    """

    def get_prefix(prefix):
        return prefix if prefix == '' else prefix + '.'
    if hasattr(mod, 'activation_post_process'):
        target_dict[get_prefix(prefix) + 'activation_post_process'] = mod.activation_post_process
    for name, child in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        _get_observer_dict(child, target_dict, module_prefix)