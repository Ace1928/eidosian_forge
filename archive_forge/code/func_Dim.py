import builtins
import copy
import dataclasses
import inspect
import io
import math
import pathlib
import sys
import typing
from enum import auto, Enum
from typing import (
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from torch.utils._pytree import (
from .exported_program import ExportedProgram, ModuleCallEntry, ModuleCallSignature
from .graph_signature import ExportBackwardSignature, ExportGraphSignature
def Dim(name: str, *, min: Optional[int]=None, max: Optional[int]=None):
    """
    :func:`Dim` constructs a type analogous to a named symbolic integer with a range.
    It can be used to describe multiple possible values of a dynamic tensor dimension.
    Note that different dynamic dimensions of the same tensor, or of different tensors,
    can be described by the same type.

    Args:
        name (str): Human-readable name for debugging.
        min (Optional[int]): Minimum possible value of given symbol (inclusive)
        max (Optional[int]): Maximum possible value of given symbol (inclusive)

    Returns:
        A type that can be used in dynamic shape specifications for tensors.
    """
    _min = 2 if min is None else builtins.max(min, 2)
    _max = sys.maxsize - 1 if max is None else builtins.min(max, sys.maxsize - 1)
    assert _max > _min, f'Cannot create Dim with inconsistent min={min}, max={max}'
    dim = _Dim(name, (int,), {'min': _min, 'max': _max})
    dim.__module__ = getattr(inspect.getmodule(inspect.stack()[1][0]), '__name__', '__main__')
    return dim