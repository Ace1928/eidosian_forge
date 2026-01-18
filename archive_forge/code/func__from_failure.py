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
@classmethod
def _from_failure(cls, export_exception: Exception, diagnostic_context: diagnostics.DiagnosticContext) -> Self:
    """
        Creates an instance of :class:`ONNXProgram` when the export process encounters a failure.

        In case of a failed export, this method is used to encapsulate the exception
        and associated diagnostic context within an :class:`ONNXProgram` instance for
        easier handling and debugging.

        Args:
            export_exception: The exception raised during the export process.
            diagnostic_context: The context associated with diagnostics during export.

        Returns:
            An instance of :class:`ONNXProgram` representing the failed ONNX program.
        """
    import onnx
    return ONNXProgram(onnx.ModelProto(), io_adapter.InputAdapter(), io_adapter.OutputAdapter(), diagnostic_context, export_exception=export_exception)