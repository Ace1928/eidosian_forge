from __future__ import annotations
import functools
import inspect
import traceback
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics.infra import _infra, formatter
@functools.lru_cache
def _function_source_info(fn: Callable) -> Tuple[Sequence[str], int, Optional[str]]:
    """Returns the source lines, line number, and source file path for the given function.

    Essentially, inspect.getsourcelines() and inspect.getsourcefile() combined.
    Caching is applied to reduce the performance impact of this function.
    """
    source_lines, lineno = inspect.getsourcelines(fn)
    return (source_lines, lineno, inspect.getsourcefile(fn))