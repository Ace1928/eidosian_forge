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