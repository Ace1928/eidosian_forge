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
def _make_matmul_test_all_dims_known(self, version, shape1: Sequence[int], shape2: Sequence[int]) -> None:
    expected_out_shape = np.matmul(np.arange(np.prod(shape1)).reshape(shape1), np.arange(np.prod(shape2)).reshape(shape2)).shape
    graph = self._make_graph([('x', TensorProto.FLOAT, shape1), ('y', TensorProto.FLOAT, shape2)], [make_node('MatMul', ['x', 'y'], ['z'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, expected_out_shape)], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])