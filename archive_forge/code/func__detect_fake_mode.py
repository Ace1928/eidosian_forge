from __future__ import annotations
import abc
import contextlib
import dataclasses
import difflib
import io
import logging
import sys
from typing import Any, Callable, Optional, Tuple
import torch
import torch.fx
from torch._subclasses import fake_tensor
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import diagnostics, onnxfunction_dispatcher
def _detect_fake_mode(self) -> Optional[fake_tensor.FakeTensorMode]:
    """Detect fake mode from the graph.

        Scan through all nodes in graph and their meta['val'] to detect fake mode.
        """
    fake_tensors = [node.meta.get('val') for node in self.module.graph.nodes]
    with maybe_disable_fake_tensor_mode():
        return torch._dynamo.utils.detect_fake_mode(fake_tensors)