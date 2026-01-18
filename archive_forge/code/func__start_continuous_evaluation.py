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
def _start_continuous_evaluation(self):
    """Repeatedly calls `Estimator` evaluate and export until training ends."""
    _assert_eval_spec(self._eval_spec)
    start_delay_secs = self._eval_spec.start_delay_secs
    if start_delay_secs:
        tf.compat.v1.logging.info('Waiting %f secs before starting eval.', start_delay_secs)
        time.sleep(start_delay_secs)
    latest_eval_result = None
    evaluator = _TrainingExecutor._Evaluator(self._estimator, self._eval_spec, self._train_spec.max_steps)
    should_early_stop = False
    while not should_early_stop:
        if latest_eval_result and latest_eval_result.status == _EvalStatus.EVALUATED:
            global_step = latest_eval_result.metrics.get(tf.compat.v1.GraphKeys.GLOBAL_STEP)
            if global_step and self._train_spec.max_steps and (global_step >= self._train_spec.max_steps):
                tf.compat.v1.logging.info('Exiting evaluation, global_step=%s >= train max_steps=%s', global_step, self._train_spec.max_steps)
                return
        latest_eval_result, should_early_stop = self._execute_evaluator_once(evaluator, self._continuous_eval_listener, self._eval_spec.throttle_secs)