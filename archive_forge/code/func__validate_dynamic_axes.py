from __future__ import annotations
import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import (
@_beartype.beartype
def _validate_dynamic_axes(dynamic_axes, model, input_names, output_names):
    """Ensures dynamic axes argument is follows the expected format."""
    if len(dynamic_axes) == 0:
        return
    if hasattr(model, 'graph'):
        if input_names is None or len(input_names) == 0:
            input_names = [x.debugName() for x in model.graph.inputs()]
        if output_names is None or len(output_names) == 0:
            output_names = [y.debugName() for y in model.graph.outputs()]
    valid_names = set((input_names or []) + (output_names or []))
    for key, value in dynamic_axes.items():
        if key not in valid_names:
            warnings.warn(f'Provided key {key} for dynamic axes is not a valid input/output name')
        if isinstance(value, list):
            warnings.warn(f'No names were found for specified dynamic axes of provided input.Automatically generated names will be applied to each dynamic axes of input {key}')
            value_dict = {}
            for i, x in enumerate(value):
                if not isinstance(x, int):
                    raise ValueError('The type of axis index is expected to be an integer')
                if x in value_dict:
                    warnings.warn(f'Duplicate dynamic axis index {x} was provided for input {key}.')
                else:
                    value_dict[x] = str(key) + '_dynamic_axes_' + str(i + 1)
            dynamic_axes[key] = value_dict