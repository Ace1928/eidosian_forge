import warnings
from contextlib import contextmanager
from typing import Any, Iterator
import torch._C
from torch._jit_internal import (
from torch.jit._async import fork, wait
from torch.jit._await import _awaitable, _awaitable_nowait, _awaitable_wait
from torch.jit._decomposition_utils import _register_decomposition
from torch.jit._freeze import freeze, optimize_for_inference, run_frozen_optimizations
from torch.jit._fuser import (
from torch.jit._ir_utils import _InsertPoint
from torch.jit._script import (
from torch.jit._serialization import (
from torch.jit._trace import (
from torch.utils import set_module
class strict_fusion:
    """
    Give errors if not all nodes have been fused in inference, or symbolically differentiated in training.

    Example:
    Forcing fusion of additions.

    .. code-block:: python

        @torch.jit.script
        def foo(x):
            with torch.jit.strict_fusion():
                return x + x + x

    """

    def __init__(self):
        if not torch._jit_internal.is_scripting():
            warnings.warn('Only works in script mode')
        pass

    def __enter__(self):
        pass

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        pass