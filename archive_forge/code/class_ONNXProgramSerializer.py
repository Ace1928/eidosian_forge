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
@runtime_checkable
class ONNXProgramSerializer(Protocol):
    """Protocol for serializing an ONNX graph into a specific format (e.g. Protobuf).
    Note that this is an advanced usage scenario."""

    def serialize(self, onnx_program: ONNXProgram, destination: io.BufferedIOBase) -> None:
        """Protocol method that must be implemented for serialization.

        Args:
            onnx_program: Represents the in-memory exported ONNX model
            destination: A binary IO stream or pre-allocated buffer into which
                the serialized model should be written.

        Example:

            A simple serializer that writes the exported :py:obj:`onnx.ModelProto` in Protobuf
            format to ``destination``:

            ::

                # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
                >>> import io
                >>> import torch
                >>> import torch.onnx
                >>> class MyModel(torch.nn.Module):  # Dummy model
                ...     def __init__(self) -> None:
                ...         super().__init__()
                ...         self.linear = torch.nn.Linear(2, 2)
                ...     def forward(self, x):
                ...         out = self.linear(x)
                ...         return out
                >>> class ProtobufONNXProgramSerializer:
                ...     def serialize(
                ...         self, onnx_program: torch.onnx.ONNXProgram, destination: io.BufferedIOBase
                ...     ) -> None:
                ...         destination.write(onnx_program.model_proto.SerializeToString())
                >>> model = MyModel()
                >>> arg1 = torch.randn(2, 2, 2)  # positional input 1
                >>> torch.onnx.dynamo_export(model, arg1).save(
                ...     destination="exported_model.onnx",
                ...     serializer=ProtobufONNXProgramSerializer(),
                ... )
        """
        ...