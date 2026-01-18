from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import os
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _get_eval_session_master(task_type, tf_config):
    """Returns the appropriate address for TensorFlow evaluation master."""
    if task_type == TaskType.EVALUATOR:
        return tf_config.get(_EVAL_SESSION_MASTER_KEY, _LOCAL_MASTER)
    return _LOCAL_MASTER