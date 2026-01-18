import functools
import os
import tempfile
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy as mwms_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import mirrored_strategy as mirrored_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_util
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
class OneDeviceDistributionTestBase(test.TestCase):
    """Some tests that should work with any one-device DistributionStrategy."""

    def _test_run(self, strategy):
        out1 = strategy.run(lambda: array_ops.identity(4.0))
        self.assertAllEqual([4.0], self.evaluate(strategy.unwrap(out1)))
        out2 = strategy.run(lambda x: {'a': x * 2, 'b': x * x}, args=(out1,))
        out2_vals = self.evaluate(nest.map_structure(strategy.unwrap, out2))
        self.assertAllEqual([8.0], out2_vals['a'])
        self.assertAllEqual([16.0], out2_vals['b'])
        out3 = strategy.run(lambda b, a: a + 2 * b + 2, kwargs=out2)
        self.assertAllEqual([42.0], self.evaluate(strategy.unwrap(out3)))

    def _test_all_reduce_sum(self, strategy):
        self._test_collective_comms(strategy, _all_sum, inputs=(4.0, [42.0, 43.0]), expected=(4.0, [42.0, 43.0]))

    def _test_all_reduce_sum_gradients(self, strategy):
        self._test_collective_comms_gradients(strategy, _all_sum, inputs=[4.0], expected_grads=[4.0])

    def _test_all_reduce_sum_gradient_tape(self, strategy):
        self._test_collective_comms_gradient_tape(strategy, _all_sum, inputs=[4.0], expected_grads=[4.0])

    def _test_all_reduce_mean(self, strategy):
        self._test_collective_comms(strategy, _all_mean, inputs=(2.0, [21.0, 22.0]), expected=(2.0, [21.0, 22.0]))

    def _test_all_reduce_mean_gradients(self, strategy):
        self._test_collective_comms_gradients(strategy, _all_mean, inputs=[5.0], expected_grads=[5.0])

    def _test_all_reduce_mean_gradient_tape(self, strategy):
        self._test_collective_comms_gradient_tape(strategy, _all_mean, inputs=[5.0], expected_grads=[5.0])

    def _test_collective_comms(self, strategy, comm_fn, inputs, expected):
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensors(inputs))
        self.evaluate(inputs.initialize())
        outputs = self.evaluate(list(map(strategy.experimental_local_results, strategy.experimental_run(comm_fn, inputs))))
        self.assertAllEqual([expected[0]], outputs[0])
        self.assertAllEqual([expected[1]], outputs[1])

    def _test_collective_comms_gradients(self, strategy, comm_fn, inputs, expected_grads):
        if context.executing_eagerly():
            self.skipTest('`tf.gradients` is not supported with eager execution.')

        def step(c):
            x = array_ops.identity(42.0)
            y = comm_fn(x) * c
            return gradients_impl.gradients(y, [x])[0]
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensors(inputs))
        self.evaluate(inputs.initialize())
        self.assertAllEqual(expected_grads, self.evaluate(strategy.experimental_local_results(strategy.experimental_run(step, inputs))))

    def _test_collective_comms_gradient_tape(self, strategy, comm_fn, inputs, expected_grads):

        def step(c):
            x = array_ops.identity(42.0)
            with backprop.GradientTape() as tape:
                tape.watch(x)
                y = comm_fn(x) * c
            return tape.gradient(y, x)
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensors(inputs))
        self.evaluate(inputs.initialize())
        self.assertAllEqual(expected_grads, self.evaluate(strategy.experimental_local_results(strategy.experimental_run(step, inputs))))

    def _test_device_and_input_device_are_colocated(self, strategy):
        if context.executing_eagerly():
            self.skipTest('cross-device tests are not supported with eager execution.')
        workers, _ = test_util.create_local_cluster(2, 0)
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.range(5))
        comm_fn = lambda x: x + 1
        run_op = strategy.experimental_run(comm_fn, inputs)
        with session_lib.Session(target=workers[1].target) as sess:
            sess.run(inputs.initialize())
            sess.run(run_op)

    def _test_device_and_input_device_are_colocated_with_function(self, strategy):
        if context.executing_eagerly():
            self.skipTest('cross-device tests are not supported with eager execution.')
        workers, _ = test_util.create_local_cluster(2, 0)
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.range(5))
        comm_fn = lambda x: x + 1
        experimental_run = def_function.function()(strategy.experimental_run)
        with ops.device('/job:worker/replica:0/task:1/device:CPU:0'):
            run_op = experimental_run(comm_fn, inputs)
        with session_lib.Session(target=workers[1].target) as sess:
            sess.run(inputs.initialize())
            sess.run(run_op)