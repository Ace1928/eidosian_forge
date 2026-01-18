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
def _get_default_warm_start_settings(warm_start_from):
    """Returns default `tf.estimator.WarmStartSettings`.

  Args:
    warm_start_from: Either a string representing the filepath of a checkpoint
      or `SavedModel` to initialize from, or an instance of
      `tf.estimator.WarmStartSettings`.

  Returns:
    Either None or an instance of `WarmStartSettings`.

  Raises:
    ValueError: If `warm_start_from` is not `None` but is neither a string nor
    an instance of `WarmStartSettings`.
  """
    if warm_start_from is None:
        return None
    if isinstance(warm_start_from, (six.string_types, six.binary_type)):
        if tf.compat.v1.gfile.Exists(os.path.join(path_helpers.get_variables_dir(warm_start_from), tf.compat.as_text('variables.index'))):
            tf.compat.v1.logging.info('Warm-starting from a SavedModel')
            return WarmStartSettings(ckpt_to_initialize_from=path_helpers.get_variables_path(warm_start_from))
        return WarmStartSettings(ckpt_to_initialize_from=warm_start_from)
    elif isinstance(warm_start_from, WarmStartSettings):
        return warm_start_from
    else:
        raise ValueError('warm_start_from must be a string or a WarmStartSettings, instead got {}'.format(type(warm_start_from)))