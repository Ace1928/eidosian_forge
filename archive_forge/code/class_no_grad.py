from typing import Any, Optional
import torch
from torch.utils._contextlib import (
class no_grad(_NoParamDecoratorContextManager):
    """Context-manager that disables gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.
    There is an exception! All factory functions, or functions that create
    a new Tensor and take a requires_grad kwarg, will NOT be affected by
    this mode.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        No-grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.
        If you want to disable forward AD for a computation, you can unpack
        your dual tensors.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> @torch.no_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> z = doubler(x)
        >>> z.requires_grad
        False
        >>> @torch.no_grad
        ... def tripler(x):
        ...     return x * 3
        >>> z = tripler(x)
        >>> z.requires_grad
        False
        >>> # factory function exception
        >>> with torch.no_grad():
        ...     a = torch.nn.Parameter(torch.rand(10))
        >>> a.requires_grad
        True
    """

    def __init__(self) -> None:
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self.prev = False

    def __enter__(self) -> None:
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.set_grad_enabled(self.prev)