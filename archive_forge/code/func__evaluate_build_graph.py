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
def _evaluate_build_graph(self, input_fn, hooks=None, checkpoint_path=None):
    """Builds the graph and related hooks to run evaluation."""
    tf.compat.v1.random.set_random_seed(self._config.tf_random_seed)
    self._create_and_assert_global_step(tf.compat.v1.get_default_graph())
    if self._eval_distribution:
        scaffold, evaluation_hooks, input_hooks, update_op, eval_dict = self._call_model_fn_eval_distributed(input_fn, self.config)
    else:
        scaffold, evaluation_hooks, input_hooks, update_op, eval_dict = self._call_model_fn_eval(input_fn, self.config)
    global_step_tensor = tf.compat.v1.train.get_global_step(tf.compat.v1.get_default_graph())
    self._maybe_warm_start(checkpoint_path)
    if tf.compat.v1.GraphKeys.GLOBAL_STEP in eval_dict:
        raise ValueError('Metric with name `global_step` is not allowed, because Estimator already defines a default metric with the same name.')
    eval_dict[tf.compat.v1.GraphKeys.GLOBAL_STEP] = global_step_tensor
    all_hooks = list(input_hooks)
    all_hooks.extend(hooks)
    all_hooks.extend(list(evaluation_hooks or []))
    if scaffold and scaffold.local_init_op:
        evaluation._get_or_create_eval_step()
        scaffold = tf.compat.v1.train.Scaffold(local_init_op=tf.group(scaffold.local_init_op, tf.compat.v1.train.Scaffold.default_local_init_op()), copy_from_scaffold=scaffold)
    return (scaffold, update_op, eval_dict, all_hooks)