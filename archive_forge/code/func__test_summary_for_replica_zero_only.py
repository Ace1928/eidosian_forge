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
def _test_summary_for_replica_zero_only(self, d):
    logdir = tempfile.mkdtemp()

    def run_fn():
        """Function executed for each replica."""
        with summary_writer.as_default():
            replica_id = distribute_lib.get_replica_context().replica_id_in_sync_group
            return summary_ops.write('a', replica_id)
    with self.cached_session() as sess, d.scope(), summary_ops.always_record_summaries():
        global_step = training_util.get_or_create_global_step()
        if not context.executing_eagerly():
            global_step.initializer.run()
        summary_ops.set_step(0)
        summary_writer = summary_ops.create_file_writer(logdir)
        output = d.extended.call_for_each_replica(run_fn)
        unwrapped = d.unwrap(output)
        if not context.executing_eagerly():
            sess.run(summary_writer.init())
            sess.run(unwrapped)
            sess.run(summary_writer.close())
        events = _events_from_logdir(self, logdir)
        self.assertLen(events, 2)
        self.assertEqual(events[1].summary.value[0].tag, 'a')
        self.assertEqual(events[1].summary.value[0].simple_value, 0.0)