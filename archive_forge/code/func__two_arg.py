from __future__ import annotations
from functools import wraps
from builtins import all as builtin_all, any as builtin_any
from ..common._aliases import (UniqueAllResult, UniqueCountsResult,
from .._internal import get_xp
import torch
from typing import TYPE_CHECKING
def _two_arg(f):

    @wraps(f)
    def _f(x1, x2, /, **kwargs):
        x1, x2 = _fix_promotion(x1, x2)
        return f(x1, x2, **kwargs)
    if _f.__doc__ is None:
        _f.__doc__ = f'Array API compatibility wrapper for torch.{f.__name__}.\n\nSee the corresponding PyTorch documentation and/or the array API specification\nfor more details.\n\n'
    return _f