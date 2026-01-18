import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def _test_short_time_fourier_transform(self, operator_name: str) -> None:
    signal = helper.make_tensor('a', TensorProto.FLOAT, dims=[2, 64], vals=np.random.rand(2, 64).astype(np.float32))
    frame_step = helper.make_tensor('b', TensorProto.INT64, dims=[1], vals=np.array([8]))
    window = helper.make_tensor('c', TensorProto.FLOAT, dims=[16], vals=np.ones(16).astype(np.float32))
    self._test_op_upgrade(operator_name, 17, [[2, 64], [1], [16]], [[2, 7, 16, 2]], [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT, TensorProto.INT64], initializer=[signal, frame_step, window])
    signal = helper.make_tensor('a', TensorProto.FLOAT, dims=[2, 64], vals=np.random.rand(2, 64).astype(np.float32))
    frame_step = helper.make_tensor('b', TensorProto.INT64, dims=[1], vals=np.array([8]))
    window = helper.make_tensor('c', TensorProto.FLOAT, dims=[16], vals=np.ones(16).astype(np.float32))
    self._test_op_upgrade(operator_name, 17, [[2, 64], [1], [16]], [[2, 7, 9, 2]], [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT, TensorProto.INT64], attrs={'onesided': 1}, initializer=[signal, frame_step, window])
    signal = helper.make_tensor('a', TensorProto.FLOAT, dims=[2, 64, 2], vals=np.random.rand(2, 64, 2).astype(np.float32))
    frame_step = helper.make_tensor('b', TensorProto.INT64, dims=[1], vals=np.array([8]))
    window = helper.make_tensor('c', TensorProto.FLOAT, dims=[16], vals=np.ones(16).astype(np.float32))
    self._test_op_upgrade(operator_name, 17, [[2, 64, 2], [1], [16]], [[2, 7, 16, 2]], [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT, TensorProto.INT64], initializer=[signal, frame_step, window])
    signal = helper.make_tensor('a', TensorProto.FLOAT, dims=[2, 64, 2], vals=np.random.rand(2, 64, 2).astype(np.float32))
    frame_step = helper.make_tensor('b', TensorProto.INT64, dims=[1], vals=np.array([8]))
    window = helper.make_tensor('c', TensorProto.FLOAT, dims=[16], vals=np.ones(16).astype(np.float32))
    frame_length = helper.make_tensor('e', TensorProto.INT64, dims=[1], vals=np.array([16]))
    self._test_op_upgrade(operator_name, 17, [[2, 64, 2], [1], [16]], [[2, 7, 9, 2]], [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT, TensorProto.INT64], attrs={'onesided': 1}, initializer=[signal, frame_step, window, frame_length])