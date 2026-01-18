from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
def _identity_prop(self, op: str, **kwargs: Any) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (30, 4, 5))], [make_node(op, 'x', 'y', **kwargs)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (30, 4, 5))])