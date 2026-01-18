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
def _assert_fake_tensor_mode(self):
    """Asserts that the model and its input do not contain fake tensors."""
    has_any_fake_tensor = pytree.tree_any(lambda x: isinstance(x, torch._subclasses.FakeTensor), (self.model_args, self.model_kwargs))
    has_any_fake_param_or_buffer = False
    if isinstance(self.model, torch.nn.Module):
        has_any_fake_param_or_buffer = pytree.tree_any(lambda x: isinstance(x, torch._subclasses.FakeTensor), (self.model.parameters(), self.model.buffers()))
    if (has_any_fake_tensor or has_any_fake_param_or_buffer) and (not self.options.fake_context):
        raise RuntimeError('Cannot export a model with fake inputs/weights without enabling fake mode.')
    has_any_non_fake_tensors = pytree.tree_any(lambda x: isinstance(x, torch.Tensor) and (not isinstance(x, torch._subclasses.FakeTensor)), (self.model_args, self.model_kwargs))
    has_any_non_fake_param_or_buffer = False
    if isinstance(self.model, torch.nn.Module):
        has_any_non_fake_param_or_buffer = pytree.tree_any(lambda x: isinstance(x, torch.Tensor) and (not isinstance(x, torch._subclasses.FakeTensor)), (self.model.parameters(), self.model.buffers()))
    if (has_any_non_fake_tensors or has_any_non_fake_param_or_buffer) and self.options.fake_context:
        raise RuntimeError('Cannot export a model with non fake inputs/weights and enabled fake mode.')