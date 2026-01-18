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
def _train_model_default(self, input_fn, hooks, saving_listeners):
    """Initiate training with `input_fn`, without `DistributionStrategies`.

    Args:
      input_fn: A function that provides input data for training as minibatches.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the training loop.
      saving_listeners: list of `tf.train.CheckpointSaverListener` objects. Used
        for callbacks that run immediately before or after checkpoint savings.

    Returns:
      Loss from training
    """
    worker_hooks = []
    with tf.Graph().as_default() as g, g.device(self._device_fn):
        tf.compat.v1.random.set_random_seed(self._config.tf_random_seed)
        global_step_tensor = self._create_and_assert_global_step(g)
        if global_step_tensor is not None:
            training_util._get_or_create_global_step_read(g)
        features, labels, input_hooks = self._get_features_and_labels_from_input_fn(input_fn, ModeKeys.TRAIN)
        worker_hooks.extend(input_hooks)
        estimator_spec = self._call_model_fn(features, labels, ModeKeys.TRAIN, self.config)
        global_step_tensor = tf.compat.v1.train.get_global_step(g)
        return self._train_with_estimator_spec(estimator_spec, worker_hooks, hooks, global_step_tensor, saving_listeners)