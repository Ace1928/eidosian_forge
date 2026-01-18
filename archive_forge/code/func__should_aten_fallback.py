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
def _should_aten_fallback(name: str, opset_version: int, operator_export_type: _C_onnx.OperatorExportTypes):
    is_exportable_aten_op = registration.registry.is_registered_op(name, opset_version)
    is_onnx_aten_export = operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN
    is_aten_fallback_export = operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    is_caffe2_build = _C_onnx._CAFFE2_ATEN_FALLBACK
    if not name.startswith('aten::'):
        return False
    if is_caffe2_build:
        if (is_onnx_aten_export or is_aten_fallback_export) and (not is_exportable_aten_op):
            return True
    elif is_onnx_aten_export or (is_aten_fallback_export and (not is_exportable_aten_op)):
        return True
    return False