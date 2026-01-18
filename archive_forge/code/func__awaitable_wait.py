import torch
from torch._jit_internal import _Await
from torch.jit._builtins import _register_builtin
from torch.utils import set_module
def _awaitable_wait(aw):
    """Request await the result of execution, if Await is not completed yet, the func will be called immediately."""
    return torch._C._awaitable_wait(aw)