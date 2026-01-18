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
@contextmanager
def _hide_source_ranges() -> Iterator[None]:
    old_enable_source_ranges = torch._C.Graph.global_print_source_ranges
    try:
        torch._C.Graph.set_global_print_source_ranges(False)
        yield
    finally:
        torch._C.Graph.set_global_print_source_ranges(old_enable_source_ranges)