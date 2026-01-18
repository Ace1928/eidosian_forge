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
@dataclasses.dataclass
class ONNXFakeContext:
    """A dataclass used to store context for model export using FakeTensor.

    This dataclass stores the FakeTensorMode instance used to convert
    real tensors and model parameters into fake tensors. This :attr:`ONNXFakeContext.fake_mode` is
    reused internally during tracing of a :class:`torch.nn.Module` into a FX :class:`GraphModule`.
    """
    fake_mode: fake_tensor.FakeTensorMode
    'The fake tensor mode used for tracing model using fake tensors and parameters.'
    state_dict_paths: Optional[Tuple[Union[str, io.BytesIO]]] = None
    'List of paths of files that contain the model :meth:`state_dict`'