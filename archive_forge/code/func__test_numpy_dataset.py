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
def _test_numpy_dataset(self, strategy, session=None, run_in_function=False):
    if not isinstance(strategy, distribute_lib.StrategyV1):
        self.skipTest('n/a: V1 only')
    cached_session = session or self.cached_session()
    with strategy.scope(), cached_session as sess:
        x = np.asarray([[1, 2], [6, 12], [2, 4], [5, 10], [3, 6], [4, 8]])
        y = np.asarray([5, 4, 3, 2, 1, 0])
        batch_size = 6
        if not strategy.extended._global_batch_size:
            batch_size = batch_size // strategy.num_replicas_in_sync
        ds = strategy.extended.experimental_make_numpy_dataset((x, y), session=sess or self.cached_session())
        ds = ds.repeat(2)
        drop_remainder = strategy.extended.experimental_require_static_shapes
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        i = strategy.make_dataset_iterator(ds)
        self.evaluate(i.initializer)

        def run_and_concatenate(strategy, i):
            x, y = strategy.experimental_run(_maybe_run_in_function(lambda z: z, run_in_function), i)
            x, y = self.evaluate((strategy.experimental_local_results(x), strategy.experimental_local_results(y)))
            return (np.concatenate(x), np.concatenate(y))
        x_1, y_1 = run_and_concatenate(strategy, i)
        self.assertAllEqual(x, x_1)
        self.assertAllEqual(y, y_1)
        x_2, y_2 = run_and_concatenate(strategy, i)
        self.assertAllEqual(x, x_2)
        self.assertAllEqual(y, y_2)
        with self.assertRaises(errors.OutOfRangeError):
            run_and_concatenate(strategy, i)