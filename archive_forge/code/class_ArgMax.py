import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
class ArgMax(Base):

    @staticmethod
    def export_no_keepdims() -> None:
        data = np.array([[2, 1], [3, 10]], dtype=np.float32)
        axis = 1
        keepdims = 0
        node = onnx.helper.make_node('ArgMax', inputs=['data'], outputs=['result'], axis=axis, keepdims=keepdims)
        result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_no_keepdims_example')
        data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
        result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_no_keepdims_random')

    @staticmethod
    def export_keepdims() -> None:
        data = np.array([[2, 1], [3, 10]], dtype=np.float32)
        axis = 1
        keepdims = 1
        node = onnx.helper.make_node('ArgMax', inputs=['data'], outputs=['result'], axis=axis, keepdims=keepdims)
        result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_keepdims_example')
        data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
        result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_keepdims_random')

    @staticmethod
    def export_default_axes_keepdims() -> None:
        data = np.array([[2, 1], [3, 10]], dtype=np.float32)
        keepdims = 1
        node = onnx.helper.make_node('ArgMax', inputs=['data'], outputs=['result'], keepdims=keepdims)
        result = argmax_use_numpy(data, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_default_axis_example')
        data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
        result = argmax_use_numpy(data, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_default_axis_random')

    @staticmethod
    def export_negative_axis_keepdims() -> None:
        data = np.array([[2, 1], [3, 10]], dtype=np.float32)
        axis = -1
        keepdims = 1
        node = onnx.helper.make_node('ArgMax', inputs=['data'], outputs=['result'], axis=axis, keepdims=keepdims)
        result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_negative_axis_keepdims_example')
        data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
        result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_negative_axis_keepdims_random')

    @staticmethod
    def export_no_keepdims_select_last_index() -> None:
        data = np.array([[2, 2], [3, 10]], dtype=np.float32)
        axis = 1
        keepdims = 0
        node = onnx.helper.make_node('ArgMax', inputs=['data'], outputs=['result'], axis=axis, keepdims=keepdims, select_last_index=True)
        result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_no_keepdims_example_select_last_index')
        data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
        result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_no_keepdims_random_select_last_index')

    @staticmethod
    def export_keepdims_select_last_index() -> None:
        data = np.array([[2, 2], [3, 10]], dtype=np.float32)
        axis = 1
        keepdims = 1
        node = onnx.helper.make_node('ArgMax', inputs=['data'], outputs=['result'], axis=axis, keepdims=keepdims, select_last_index=True)
        result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_keepdims_example_select_last_index')
        data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
        result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_keepdims_random_select_last_index')

    @staticmethod
    def export_default_axes_keepdims_select_last_index() -> None:
        data = np.array([[2, 2], [3, 10]], dtype=np.float32)
        keepdims = 1
        node = onnx.helper.make_node('ArgMax', inputs=['data'], outputs=['result'], keepdims=keepdims, select_last_index=True)
        result = argmax_use_numpy_select_last_index(data, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_default_axis_example_select_last_index')
        data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
        result = argmax_use_numpy_select_last_index(data, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_default_axis_random_select_last_index')

    @staticmethod
    def export_negative_axis_keepdims_select_last_index() -> None:
        data = np.array([[2, 2], [3, 10]], dtype=np.float32)
        axis = -1
        keepdims = 1
        node = onnx.helper.make_node('ArgMax', inputs=['data'], outputs=['result'], axis=axis, keepdims=keepdims, select_last_index=True)
        result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_negative_axis_keepdims_example_select_last_index')
        data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
        result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmax_negative_axis_keepdims_random_select_last_index')