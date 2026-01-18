import torch
from torch._jit_internal import Future
from torch.jit._builtins import _register_builtin
from torch.utils import set_module

    Force completion of a `torch.jit.Future[T]` asynchronous task, returning the result of the task.

    See :func:`~fork` for docs and examples.
    Args:
        future (torch.jit.Future[T]): an asynchronous task reference, created through `torch.jit.fork`
    Returns:
        `T`: the return value of the completed task
    