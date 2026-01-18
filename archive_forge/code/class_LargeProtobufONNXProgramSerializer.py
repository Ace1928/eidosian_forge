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
class LargeProtobufONNXProgramSerializer:
    """Serializes ONNX graph as Protobuf.

    Fallback to serializing as Protobuf with external data for models larger than 2GB.
    """
    _destination_path: Final[str]

    def __init__(self, destination_path: str):
        self._destination_path = destination_path

    @_beartype.beartype
    def serialize(self, onnx_program: ONNXProgram, destination: io.BufferedIOBase) -> None:
        """`destination` is ignored. The model is saved to `self._destination_path` instead."""
        import onnx
        if onnx_program.model_proto.ByteSize() < _PROTOBUF_SIZE_MAX_LIMIT:
            onnx.save_model(onnx_program.model_proto, self._destination_path)
        else:
            onnx.save_model(onnx_program.model_proto, self._destination_path, save_as_external_data=True, all_tensors_to_one_file=True)