import collections
import io
import sys
import types
from typing import (
import torch
import torch.distributed.rpc as rpc
from torch import Tensor, device, dtype, nn
from torch.distributed.nn.jit import instantiator
from torch.distributed import _remote_device
from torch.distributed.rpc.internal import _internal_rpc_pickler
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
def _create_module_with_interface(module_cls, args, kwargs, device, module_interface_cls):
    module = _create_module(module_cls, args, kwargs, device)
    if module_interface_cls is not None:
        module = torch.jit.script(module)
    return rpc.RRef(module, module_interface_cls)