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
def _logical_binary_op_with_broadcasting(self, op: str, input_type: TensorProto.DataType) -> None:
    graph = self._make_graph([('x', input_type, (1, 5)), ('y', input_type, (30, 4, 5))], [make_node(op, ['x', 'y'], 'z')], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.BOOL, (30, 4, 5))])