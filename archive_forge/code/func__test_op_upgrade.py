import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def _test_op_upgrade(self, op, *args, **kwargs):
    self.tested_ops.append(op)
    self._test_op_conversion(op, *args, **kwargs, is_upgrade=True)