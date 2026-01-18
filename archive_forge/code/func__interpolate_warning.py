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
def _interpolate_warning(interpolate_mode):
    onnx_op = 'onnx:Resize' if GLOBALS.export_onnx_opset_version >= 10 else 'onnx:Upsample'
    warnings.warn('You are trying to export the model with ' + onnx_op + ' for ONNX opset version ' + str(GLOBALS.export_onnx_opset_version) + ". This operator might cause results to not match the expected results by PyTorch.\nONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11. Attributes to determine how to transform the input were added in onnx:Resize in opset 11 to support Pytorch's behavior (like coordinate_transformation_mode and nearest_mode).\nWe recommend using opset 11 and above for models using this operator.")