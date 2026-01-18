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
def _actual_eval(self, input_fn, strategy=None, steps=None, hooks=None, checkpoint_path=None, name=None):
    """The method that does evaluation actually."""
    with context.graph_mode():
        hooks = _check_hooks_type(hooks)
        hooks.extend(self._convert_eval_steps_to_hooks(steps))
        if not checkpoint_path:
            latest_path = checkpoint_management.latest_checkpoint(self._model_dir)
            if not latest_path:
                tf.compat.v1.logging.info('Could not find trained model in model_dir: {}, running initialization to evaluate.'.format(self._model_dir))
            checkpoint_path = latest_path

        def _evaluate():
            scaffold, update_op, eval_dict, all_hooks = self._evaluate_build_graph(input_fn, hooks, checkpoint_path)
            return self._evaluate_run(checkpoint_path=checkpoint_path, scaffold=scaffold, update_op=update_op, eval_dict=eval_dict, all_hooks=all_hooks, output_dir=self.eval_dir(name))
        with tf.Graph().as_default():
            if strategy:
                training.get_or_create_steps_per_run_variable()
                with strategy.scope():
                    return _evaluate()
            else:
                return _evaluate()