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
def _validate_saving_listeners(saving_listeners):
    """Validates the `saving_listeners`."""
    saving_listeners = tuple(saving_listeners or [])
    for saving_listener in saving_listeners:
        if not isinstance(saving_listener, tf.compat.v1.train.CheckpointSaverListener):
            raise TypeError('All saving_listeners must be `CheckpointSaverListener` instances, given: {}'.format(saving_listener))
    return saving_listeners