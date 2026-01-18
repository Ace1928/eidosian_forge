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
def _train_with_estimator_spec_distributed(self, estimator_spec, worker_hooks, saving_listener):
    """Train a model with the given Estimator Spec and Distribution Strategy."""
    if saving_listener:
        raise ValueError('Saving listenor is not supported by the current Distribution Strategies.')
    with training.MonitoredTrainingSession(master=self._config.master, is_chief=self._config.is_chief, checkpoint_dir=self._model_dir, scaffold=estimator_spec.scaffold, hooks=worker_hooks, chief_only_hooks=tuple(estimator_spec.training_chief_hooks), save_checkpoint_secs=self._config.save_checkpoints_secs, save_checkpoint_steps=self._config.save_checkpoints_steps, save_summaries_steps=self._config.save_summary_steps, config=self._session_config, max_wait_secs=self._config.session_creation_timeout_secs, log_step_count_steps=self._config.log_step_count_steps, save_graph_def=self._config.checkpoint_save_graph_def) as mon_sess:
        loss = None
        current_step = 0
        while not mon_sess.should_stop():
            current_step += 1
            with trace.Trace('train', step_num=current_step, _r=1):
                _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
        if current_step == 0:
            tf.compat.v1.logging.warn('Training with estimator made no steps. Perhaps input is empty or misspecified.')
    return loss