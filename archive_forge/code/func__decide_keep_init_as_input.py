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
def _decide_keep_init_as_input(keep_initializers_as_inputs: Optional[bool], operator_export_type: _C_onnx.OperatorExportTypes, opset_version: int):
    """Decides whether the initializers in the graph should be listed as ONNX graph inputs.

    This method encapsulates the logic to decide whether the initializers in the graph
    should be listed as ONNX graph inputs (i.e., whether to choose ONNX IR v3 or v4).
    If keep_initializers_as_inputs is not specified (None), then we decide whether to keep
    initializers as graph inputs (val_keep_init_as_ip) based on export type. If export type
    is ONNX, then do not keep initializers as input (val_keep_init_as_ip=False). For all other
    export types keep initializers as input (val_keep_init_as_ip=True).
    If keep_initializers_as_inputs is specified, then respect it. Unless opset version <= 8,
    in which case it must be ignored because for opset version <= 8, all initializers MUST be
    part of graph input (only ONNX IR v3 is allowed), i.e. val_keep_init_as_ip=True.

    Special handling is needed for opset version 8 or lower, because irrespective
    of user input for keep_initializers_as_inputs, the graph must follow ONNX IR v3
    semantics, i.e. all initializers must be listed as ONNX graph input.
    """
    if opset_version < 9:
        if keep_initializers_as_inputs is False:
            warnings.warn("Setting 'keep_initializers_as_inputs=False' for opset version8 or lower would lead to an invalid ONNX graph. Therefore, 'keep_initializers_as_inputs=False' is ignored during export.Exported model will have initializers as graph inputs (compliant  to ONNX IR v3).")
        return True
    val_keep_init_as_ip = True if keep_initializers_as_inputs is None else keep_initializers_as_inputs
    if keep_initializers_as_inputs is None and operator_export_type is _C_onnx.OperatorExportTypes.ONNX:
        val_keep_init_as_ip = False
    return val_keep_init_as_ip