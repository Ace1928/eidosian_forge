import torch
from torch._jit_internal import _Await
from torch.jit._builtins import _register_builtin
from torch.utils import set_module
def _awaitable_nowait(o):
    """Create completed Await with specified result."""
    return torch._C._awaitable_nowait(o)