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
def _recursive_script_module_reducer(recursive_script_module):
    """Serialize a RecursiveScriptModule that does not contain a script RemoteModule, and raises an error otherwise."""
    if hasattr(recursive_script_module._c, 'module_rref'):
        raise RuntimeError('Passing a script RemoteModule over RPC is not supported. Please create a RemoteModule in the sender, send the `module_rref` to the receiver, and create a new instance on the receiver end by passing this `module_rref`.')
    f = io.BytesIO()
    torch.jit.save(recursive_script_module, f)
    return (_recursive_script_module_receiver, (f.getvalue(),))