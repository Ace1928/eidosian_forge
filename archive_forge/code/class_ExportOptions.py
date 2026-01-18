from __future__ import (  # for onnx.ModelProto (ONNXProgram) and onnxruntime (ONNXRuntimeOptions)
import abc
import contextlib
import dataclasses
import io
import logging
import os
import warnings
from collections import defaultdict
from typing import (
from typing_extensions import Self
import torch
import torch._ops
import torch.export as torch_export
import torch.utils._pytree as pytree
from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, io_adapter
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import (
class ExportOptions:
    """Options to influence the TorchDynamo ONNX exporter.

    Attributes:
        dynamic_shapes: Shape information hint for input/output tensors.
            When ``None``, the exporter determines the most compatible setting.
            When ``True``, all input shapes are considered dynamic.
            When ``False``, all input shapes are considered static.
        op_level_debug: Whether to export the model with op-level debug information
        diagnostic_options: The diagnostic options for the exporter.
        fake_context: The fake context used for symbolic tracing.
        onnx_registry: The ONNX registry used to register ATen operators to ONNX functions.
    """
    dynamic_shapes: Optional[bool] = None
    'Shape information hint for input/output tensors.\n\n    - ``None``: the exporter determines the most compatible setting.\n    - ``True``: all input shapes are considered dynamic.\n    - ``False``: all input shapes are considered static.\n    '
    op_level_debug: Optional[bool] = None
    'When True export the model with op-level debug running ops through ONNX Runtime.'
    diagnostic_options: DiagnosticOptions
    'The diagnostic options for the exporter.'
    fake_context: Optional[ONNXFakeContext] = None
    'The fake context used for symbolic tracing.'
    onnx_registry: Optional[OnnxRegistry] = None
    'The ONNX registry used to register ATen operators to ONNX functions.'

    @_beartype.beartype
    def __init__(self, *, dynamic_shapes: Optional[bool]=None, op_level_debug: Optional[bool]=None, fake_context: Optional[ONNXFakeContext]=None, onnx_registry: Optional[OnnxRegistry]=None, diagnostic_options: Optional[DiagnosticOptions]=None):
        self.dynamic_shapes = dynamic_shapes
        self.op_level_debug = op_level_debug
        self.fake_context = fake_context
        self.onnx_registry = onnx_registry
        self.diagnostic_options = diagnostic_options or DiagnosticOptions()