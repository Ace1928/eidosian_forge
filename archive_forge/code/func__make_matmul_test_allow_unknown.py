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
def _make_matmul_test_allow_unknown(self, version, shape1: Any, shape2: Any, expected_out_shape: Any) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, shape1), ('y', TensorProto.FLOAT, shape2)], [make_node('MatMul', ['x', 'y'], ['z'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, expected_out_shape)], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])