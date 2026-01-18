from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import os
import tempfile
import numpy as np
import six
import tensorflow as tf
from google.protobuf import message
from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import device_setter
from tensorflow.python.training import evaluation
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.tools.docs import doc_controls
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator import util as estimator_util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _combine_distributed_scaffold(grouped_scaffold, distribution):
    """Combines scaffold(s) returned from `call_for_each_replica`."""
    scaffold_list = distribution.experimental_local_results(grouped_scaffold)
    init_feed_dict = [s.init_feed_dict for s in scaffold_list if s.init_feed_dict is not None]
    if init_feed_dict:
        init_feed_dict = distribution.group(init_feed_dict)
    else:
        init_feed_dict = None
    init_fn = [s._user_init_fn for s in scaffold_list if s._user_init_fn is not None]
    if init_fn:
        init_fn = init_fn[0]
    else:
        init_fn = None
    init_op = [s.init_op for s in scaffold_list if s.init_op is not None]
    if init_op:
        init_op = distribution.group(init_op)
    else:
        init_op = None

    def _unwrap_and_concat(value):
        value = tf.nest.flatten(distribution.experimental_local_results(value))
        if len(value) != 1:
            return tf.concat(value, 0)
        return value[0]
    ready_op = distribution.extended.call_for_each_replica(lambda scaffold: scaffold.ready_op, args=(grouped_scaffold,))
    if ready_op is not None:
        ready_op = _unwrap_and_concat(ready_op)
    ready_for_local_init_op = distribution.extended.call_for_each_replica(create_per_replica_ready_for_local_init_op, args=(grouped_scaffold,))
    if ready_for_local_init_op is not None:
        ready_for_local_init_op = _unwrap_and_concat(ready_for_local_init_op)
    else:
        ready_for_local_init_op = None
    local_init_op = [s.local_init_op for s in scaffold_list if s.local_init_op is not None]
    if local_init_op:
        local_init_op = distribution.group(local_init_op)
    else:
        local_init_op = None
    summary_op = [s.summary_op for s in scaffold_list if s.summary_op is not None]
    if summary_op:
        summary_op = distribution.group(summary_op)
    else:
        summary_op = None
    savers = [s.saver for s in scaffold_list if s.saver is not None]
    if savers:
        saver = savers[0]
    else:
        saver = None
    scaffold = tf.compat.v1.train.Scaffold(init_op=init_op, ready_op=ready_op, ready_for_local_init_op=ready_for_local_init_op, local_init_op=local_init_op, summary_op=summary_op, saver=saver, init_feed_dict=init_feed_dict, init_fn=init_fn)
    return scaffold