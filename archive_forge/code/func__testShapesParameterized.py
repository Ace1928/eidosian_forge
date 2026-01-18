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