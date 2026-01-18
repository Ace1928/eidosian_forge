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
def _test_run(self, strategy, run_in_function=False):
    out1 = strategy.run(_maybe_run_in_function(lambda: distribute_lib.get_replica_context().replica_id_in_sync_group + 1, run_in_function))
    self.assertAllEqual([1, 2], self.evaluate(strategy.unwrap(out1)))
    out2 = strategy.run(_maybe_run_in_function(lambda x: {'a': x * 2, 'b': x * x}, run_in_function), args=(out1,))
    out2_vals = self.evaluate(nest.map_structure(strategy.unwrap, out2))
    self.assertAllEqual([2, 4], out2_vals['a'])
    self.assertAllEqual([1, 4], out2_vals['b'])
    out3 = strategy.run(_maybe_run_in_function(lambda b, a: a + 2 * b + 2, run_in_function), kwargs=out2)
    self.assertAllEqual([6, 14], self.evaluate(strategy.unwrap(out3)))