from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import time
import six
import tensorflow as tf
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import server_lib
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import exporter as exporter_lib
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _validate_hooks(hooks):
    """Validates the `hooks`."""
    hooks = tuple(hooks or [])
    for hook in hooks:
        if not isinstance(hook, tf.compat.v1.train.SessionRunHook):
            raise TypeError('All hooks must be `SessionRunHook` instances, given: {}'.format(hook))
    return hooks