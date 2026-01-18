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
def _make_qlinearmatmul_test_allow_unknown(self, shape1: Any, shape2: Any, expected_out_shape: Any) -> None:
    graph = self._make_graph([('a', TensorProto.UINT8, shape1), ('a_scale', TensorProto.FLOAT, ()), ('a_zero_point', TensorProto.UINT8, ()), ('b', TensorProto.UINT8, shape2), ('b_scale', TensorProto.FLOAT, ()), ('b_zero_point', TensorProto.UINT8, ()), ('y_scale', TensorProto.FLOAT, ()), ('y_zero_point', TensorProto.UINT8, ())], [make_node('QLinearMatMul', ['a', 'a_scale', 'a_zero_point', 'b', 'b_scale', 'b_zero_point', 'y_scale', 'y_zero_point'], ['y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, expected_out_shape)])