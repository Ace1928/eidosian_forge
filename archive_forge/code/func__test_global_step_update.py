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
def _test_global_step_update(self, strategy):
    with strategy.scope():
        global_step = variable_scope.get_variable('global_step', shape=[], dtype=dtypes.int64, initializer=init_ops.zeros_initializer(), trainable=False, aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)
        self.evaluate(variables.global_variables_initializer())

        def model_fn():
            train_op = global_step.assign_add(1)
            value = global_step.read_value()
            return (train_op, value)
        train_ops, value = strategy.extended.call_for_each_replica(model_fn)
        self.evaluate(strategy.group(train_ops))
        global_step_tensors = strategy.experimental_local_results(value)
        global_step_values = self.evaluate(global_step_tensors)
        self.assertEqual((1,) * len(global_step_tensors), global_step_values)