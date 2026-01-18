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
def _verify_and_create_loss_metric(eval_metric_ops, loss, distribution=None):
    """Creates a metric for loss and throws an error if one already exists."""
    if model_fn_lib.LOSS_METRIC_KEY in eval_metric_ops:
        raise ValueError('Metric with name "%s" is not allowed, because Estimator ' % model_fn_lib.LOSS_METRIC_KEY + 'already defines a default metric with the same name.')
    if distribution is None:
        loss_metric = tf.compat.v1.metrics.mean(loss)
    else:
        loss_metric = distribution.extended.call_for_each_replica(tf.compat.v1.metrics.mean, args=(loss,))
    eval_metric_ops[model_fn_lib.LOSS_METRIC_KEY] = loss_metric
    return eval_metric_ops