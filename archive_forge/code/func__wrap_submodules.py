from contextlib import contextmanager
import torch
import torch._custom_ops
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils import _pytree as pytree
@contextmanager
def _wrap_submodules(f, preserve_signature, module_call_signatures):
    tasks = []
    try:
        for path in preserve_signature:
            tasks.append(_wrap_submodule(f, path, module_call_signatures))
        yield
    finally:
        for submodule in tasks:
            del submodule.__dict__['forward']