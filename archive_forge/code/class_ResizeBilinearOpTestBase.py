from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
class ResizeBilinearOpTestBase(test.TestCase, parameterized.TestCase):

    def _itGen(self, smaller_shape, larger_shape):
        up_sample = (smaller_shape, larger_shape)
        down_sample = (larger_shape, smaller_shape)
        pass_through = (larger_shape, larger_shape)
        shape_pairs = (up_sample, down_sample, pass_through)
        options = [(True, False)]
        if not test_util.is_xla_enabled():
            options += [(False, True), (False, False)]
        for align_corners, half_pixel_centers in options:
            for in_shape, out_shape in shape_pairs:
                yield (in_shape, out_shape, align_corners, half_pixel_centers)

    def _getJacobians(self, in_shape, out_shape, align_corners=False, half_pixel_centers=False, dtype=np.float32, use_gpu=False, force_gpu=False):
        with self.cached_session(use_gpu=use_gpu, force_gpu=force_gpu):
            x = np.arange(np.prod(in_shape)).reshape(in_shape).astype(dtype)
            input_tensor = constant_op.constant(x, shape=in_shape)

            def func(in_tensor):
                return image_ops.resize_bilinear(in_tensor, out_shape[1:3], align_corners=align_corners, half_pixel_centers=half_pixel_centers)
            return gradient_checker_v2.compute_gradient(func, [input_tensor])

    @parameterized.parameters(set((True, context.executing_eagerly())))
    def _testShapesParameterized(self, use_tape):
        TEST_CASES = [[1, 1], [2, 3], [5, 4]]
        for batch_size, channel_count in TEST_CASES:
            smaller_shape = [batch_size, 2, 3, channel_count]
            larger_shape = [batch_size, 4, 6, channel_count]
            for in_shape, out_shape, _, _ in self._itGen(smaller_shape, larger_shape):
                with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
                    x = np.arange(np.prod(in_shape)).reshape(in_shape).astype(np.float32)
                    input_tensor = constant_op.constant(x, shape=in_shape)
                    tape.watch(input_tensor)
                    resized_tensor = image_ops.resize_bilinear(input_tensor, out_shape[1:3])
                    self.assertEqual(out_shape, list(resized_tensor.get_shape()))
                grad_tensor = tape.gradient(resized_tensor, input_tensor)
                self.assertEqual(in_shape, list(grad_tensor.get_shape()))
                with self.cached_session():
                    resized_values = self.evaluate(resized_tensor)
                    self.assertEqual(out_shape, list(resized_values.shape))
                    grad_values = self.evaluate(grad_tensor)
                    self.assertEqual(in_shape, list(grad_values.shape))

    @parameterized.parameters({'batch_size': 1, 'channel_count': 1}, {'batch_size': 4, 'channel_count': 3}, {'batch_size': 3, 'channel_count': 2})
    def testGradients(self, batch_size, channel_count):
        smaller_shape = [batch_size, 2, 3, channel_count]
        larger_shape = [batch_size, 5, 6, channel_count]
        for in_shape, out_shape, align_corners, half_pixel_centers in self._itGen(smaller_shape, larger_shape):
            jacob_a, jacob_n = self._getJacobians(in_shape, out_shape, align_corners, half_pixel_centers)
            threshold = 0.005
            self.assertAllClose(jacob_a, jacob_n, threshold, threshold)

    def testTypes(self):
        in_shape = [1, 4, 6, 1]
        out_shape = [1, 2, 3, 1]
        for use_gpu in [False, True]:
            for dtype in [np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]:
                jacob_a, jacob_n = self._getJacobians(in_shape, out_shape, dtype=dtype, use_gpu=use_gpu)
                if dtype in (np.float16, dtypes.bfloat16.as_numpy_dtype):
                    _, jacob_n = self._getJacobians(in_shape, out_shape, dtype=np.float32, use_gpu=use_gpu)
                threshold = 0.001
                if dtype == np.float64:
                    threshold = 1e-05
                self.assertAllClose(jacob_a, jacob_n, threshold, threshold)

    @parameterized.parameters(set((True, context.executing_eagerly())))
    def testGradOnUnsupportedType(self, use_tape):
        in_shape = [1, 4, 6, 1]
        out_shape = [1, 2, 3, 1]
        with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
            x = np.arange(0, 24).reshape(in_shape).astype(np.uint8)
            input_tensor = constant_op.constant(x, shape=in_shape)
            tape.watch(input_tensor)
            resize_out = image_ops.resize_bilinear(input_tensor, out_shape[1:3])
            with self.cached_session():
                grad = tape.gradient(resize_out, [input_tensor])
        self.assertEqual([None], grad)

    def _gpuVsCpuCase(self, in_shape, out_shape, align_corners, half_pixel_centers, dtype):
        grad = {}
        for use_gpu in [False, True]:
            grad[use_gpu] = self._getJacobians(in_shape, out_shape, align_corners, half_pixel_centers, dtype=dtype, use_gpu=use_gpu)
        threshold = 0.0001
        self.assertAllClose(grad[False], grad[True], rtol=threshold, atol=threshold)

    @parameterized.parameters({'batch_size': 1, 'channel_count': 1}, {'batch_size': 2, 'channel_count': 3}, {'batch_size': 5, 'channel_count': 4})
    def testCompareGpuVsCpu(self, batch_size, channel_count):
        smaller_shape = [batch_size, 4, 6, channel_count]
        larger_shape = [batch_size, 8, 16, channel_count]
        for params in self._itGen(smaller_shape, larger_shape):
            self._gpuVsCpuCase(*params, dtype=np.float32)

    def testCompareGpuVsCpuFloat64(self):
        in_shape = [1, 5, 7, 1]
        out_shape = [1, 9, 11, 1]
        self._gpuVsCpuCase(in_shape, out_shape, align_corners=True, half_pixel_centers=False, dtype=np.float64)