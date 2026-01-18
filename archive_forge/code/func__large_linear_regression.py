import os
import tempfile
import unittest
import numpy as np
import numpy.testing as npt
import onnx
import onnx.helper
import onnx.model_container
import onnx.numpy_helper
import onnx.reference
def _large_linear_regression():
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, None])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None])
    graph = onnx.helper.make_graph([onnx.helper.make_node('MatMul', ['X', 'A'], ['XA']), onnx.helper.make_node('MatMul', ['XA', 'B'], ['XB']), onnx.helper.make_node('MatMul', ['XB', 'C'], ['Y'])], 'mm', [X], [Y], [onnx.model_container.make_large_tensor_proto('#loc0', 'A', onnx.TensorProto.FLOAT, (3, 3)), onnx.numpy_helper.from_array(np.arange(9).astype(np.float32).reshape((-1, 3)), name='B'), onnx.model_container.make_large_tensor_proto('#loc1', 'C', onnx.TensorProto.FLOAT, (3, 3))])
    onnx_model = onnx.helper.make_model(graph)
    large_model = onnx.model_container.make_large_model(onnx_model.graph, {'#loc0': (np.arange(9) * 100).astype(np.float32).reshape((-1, 3)), '#loc1': (np.arange(9) + 10).astype(np.float32).reshape((-1, 3))})
    large_model.check_model()
    return large_model