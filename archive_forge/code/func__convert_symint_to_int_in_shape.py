from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import onnxscript  # type: ignore[import]
from onnxscript import evaluator  # type: ignore[import]
import torch
import torch.fx
from torch.fx.experimental import symbolic_shapes
from torch.onnx import _constants, _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from torch.utils import _pytree
@_beartype.beartype
def _convert_symint_to_int_in_shape(shape: torch.Size) -> torch.Size:
    """Convert SymInt to int in shape

    Args:
        shape (torch.Size): The shape of a tensor
    Raises:
        ValueError: When SymInt is found in shape
    Returns:
        torch.Size: The shape of a tensor with SymInt converted to int

    """
    list_int_shape = []
    for dim in shape:
        if isinstance(dim, torch.SymInt):
            if symbolic_shapes.has_hint(dim):
                list_int_shape.append(symbolic_shapes.hint_int(dim))
            else:
                raise ValueError(f'An unbacked SymInt found in shape. SymInt: {dim}; torch.Size: {shape}. There is no hint for SymInt.')
        else:
            list_int_shape.append(dim)
    return torch.Size(list_int_shape)