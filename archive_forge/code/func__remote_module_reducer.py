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
def _remote_module_reducer(remote_module):
    """Serialize a RemoteModule."""
    pickled_attrs = {}
    for k, v in remote_module.__dict__.items():
        if k == 'module_rref':
            pickled_attrs[k] = v._serialize()
        elif k in _REMOTE_MODULE_PICKLED_ATTRIBUTES:
            pickled_attrs[k] = v
        elif k not in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING:
            print(f'The new attribute ``{k}`` of RemoteModule is ignored during RPC pickling. To pickle this attribute, please add it to ``_REMOTE_MODULE_PICKLED_ATTRIBUTES``. Otherwise, please explicitly add it to ``_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING``.', file=sys.stderr)
    return (_remote_module_receiver, tuple(pickled_attrs.values()))