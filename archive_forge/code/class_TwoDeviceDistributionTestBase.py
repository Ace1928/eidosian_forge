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
class TwoDeviceDistributionTestBase(test.TestCase):
    """Some tests that should work with any two-device DistributionStrategy."""

    def _test_run(self, strategy, run_in_function=False):
        out1 = strategy.run(_maybe_run_in_function(lambda: distribute_lib.get_replica_context().replica_id_in_sync_group + 1, run_in_function))
        self.assertAllEqual([1, 2], self.evaluate(strategy.unwrap(out1)))
        out2 = strategy.run(_maybe_run_in_function(lambda x: {'a': x * 2, 'b': x * x}, run_in_function), args=(out1,))
        out2_vals = self.evaluate(nest.map_structure(strategy.unwrap, out2))
        self.assertAllEqual([2, 4], out2_vals['a'])
        self.assertAllEqual([1, 4], out2_vals['b'])
        out3 = strategy.run(_maybe_run_in_function(lambda b, a: a + 2 * b + 2, run_in_function), kwargs=out2)
        self.assertAllEqual([6, 14], self.evaluate(strategy.unwrap(out3)))

    def _test_all_reduce_sum(self, strategy, run_in_function=False):
        self._test_collective_comms(strategy, _all_sum, inputs=([1.0, 3.0], [[39.0, 2.0], [3.0, 41.0]]), expected=(4.0, [42.0, 43.0]), run_in_function=run_in_function)

    def _test_all_reduce_sum_gradients(self, strategy, run_in_function=False):
        self._test_collective_comms_gradients(strategy, _all_sum, inputs=[1.0, 3.0], expected_grads=[4.0, 4.0], run_in_function=run_in_function)

    def _test_all_reduce_sum_gradient_tape(self, strategy, run_in_function=False):
        self._test_collective_comms_gradient_tape(strategy, _all_sum, inputs=[1.0, 3.0], expected_grads=[4.0, 4.0], run_in_function=run_in_function)

    def _test_all_reduce_mean(self, strategy, run_in_function=False):
        self._test_collective_comms(strategy, _all_mean, inputs=([1.0, 3.0], [[39.0, 2.0], [3.0, 41.0]]), expected=(2.0, [21.0, 21.5]), run_in_function=run_in_function)

    def _test_all_reduce_mean_gradients(self, strategy, run_in_function=False):
        self._test_collective_comms_gradients(strategy, _all_mean, inputs=[1.0, 3.0], expected_grads=[2.0, 2.0], run_in_function=run_in_function)

    def _test_all_reduce_mean_gradient_tape(self, strategy, run_in_function=False):
        self._test_collective_comms_gradient_tape(strategy, _all_mean, inputs=[1.0, 3.0], expected_grads=[2.0, 2.0], run_in_function=run_in_function)

    def _test_collective_comms(self, strategy, comm_fn, inputs, expected, run_in_function=False):
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensor_slices(inputs))
        self.evaluate(inputs.initialize())
        outputs = self.evaluate(list(map(strategy.experimental_local_results, strategy.experimental_run(_maybe_run_in_function(comm_fn, run_in_function), inputs))))
        self.assertAllEqual([expected[0], expected[0]], outputs[0])
        self.assertAllEqual([expected[1], expected[1]], outputs[1])

    def _test_collective_comms_gradients(self, strategy, comm_fn, inputs, expected_grads, run_in_function=False):
        if context.executing_eagerly() and (not run_in_function):
            self.skipTest('`tf.gradients` is not supported with eager execution without using tf.functions.')

        def step(c):
            x = array_ops.identity(42.0)
            y = comm_fn(x) * c
            return gradients_impl.gradients(y, [x])[0]
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensor_slices(inputs))
        self.evaluate(inputs.initialize())
        self.assertAllEqual(expected_grads, self.evaluate(strategy.experimental_local_results(strategy.experimental_run(_maybe_run_in_function(step, run_in_function), inputs))))

    def _test_collective_comms_gradient_tape(self, strategy, comm_fn, inputs, expected_grads, run_in_function=False):

        def step(c):
            x = array_ops.identity(42.0)
            with backprop.GradientTape() as tape:
                tape.watch(x)
                y = comm_fn(x) * c
            return tape.gradient(y, x)
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensor_slices(inputs))
        self.evaluate(inputs.initialize())
        self.assertAllEqual(expected_grads, self.evaluate(strategy.experimental_local_results(strategy.experimental_run(_maybe_run_in_function(step, run_in_function), inputs))))