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
def _start_distributed_training(self, saving_listeners=None):
    """Calls `Estimator` train in a distributed setting."""
    config = self._estimator.config
    if not _is_google_env():
        self._start_std_server(config)
    start_delay_secs = 0
    if config.task_type == run_config_lib.TaskType.WORKER:
        max_delay_secs = _MAX_DELAY_SECS
        if config.experimental_max_worker_delay_secs is not None:
            max_delay_secs = int(config.experimental_max_worker_delay_secs)
        start_delay_secs = min(max_delay_secs, (config.task_id + 1) * _DELAY_SECS_PER_WORKER)
    if start_delay_secs > 0:
        tf.compat.v1.logging.info('Waiting %d secs before starting training.', start_delay_secs)
        time.sleep(start_delay_secs)
    self._estimator.train(input_fn=self._train_spec.input_fn, max_steps=self._train_spec.max_steps, hooks=list(self._train_spec.hooks) + list(self._train_hooks), saving_listeners=saving_listeners)