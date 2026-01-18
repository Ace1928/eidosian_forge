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
def _make_matmulinteger_test(self, shape1: Sequence[int], shape2: Sequence[int]) -> None:
    expected_out_shape = np.matmul(np.arange(np.prod(shape1)).reshape(shape1), np.arange(np.prod(shape2)).reshape(shape2)).shape
    graph = self._make_graph([('A', TensorProto.UINT8, shape1), ('B', TensorProto.UINT8, shape2), ('a_zero_point', TensorProto.UINT8, ()), ('b_zero_point', TensorProto.UINT8, ())], [make_node('MatMulInteger', ['A', 'B', 'a_zero_point', 'b_zero_point'], ['Y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.INT32, expected_out_shape)])