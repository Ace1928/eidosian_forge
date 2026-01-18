from __future__ import annotations
import functools
import inspect
import sys
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _type_utils, errors
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils
from torch.types import Number
@_beartype.beartype
def _block_list_in_opset(name: str):

    def symbolic_fn(*args, **kwargs):
        raise errors.OnnxExporterError(f'ONNX export failed on {name}, which is not implemented for opset {GLOBALS.export_onnx_opset_version}. Try exporting with other opset versions.')
    return symbolic_fn