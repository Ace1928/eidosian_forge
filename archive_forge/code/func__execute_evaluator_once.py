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
def _execute_evaluator_once(self, evaluator, continuous_eval_listener, throttle_secs):
    """Executes the `evaluator`."""
    _assert_eval_spec(self._eval_spec)
    start = time.time()
    eval_result = None
    should_early_stop = False
    if not continuous_eval_listener.before_eval():
        tf.compat.v1.logging.info('Exiting evaluation, as requested by _ContinuousEvalListener.before_eval.')
        should_early_stop = True
        return (eval_result, should_early_stop)
    eval_result, _ = evaluator.evaluate_and_export()
    if not self._continuous_eval_listener.after_eval(eval_result):
        tf.compat.v1.logging.info('Exiting evaluation, as requested by _ContinuousEvalListener.after_eval.')
        should_early_stop = True
        return (eval_result, should_early_stop)
    elapsed_time = time.time() - start
    difference = throttle_secs - elapsed_time
    if difference > 0:
        tf.compat.v1.logging.info('Waiting %f secs before starting next eval run.', difference)
        time.sleep(difference)
    elif throttle_secs == 0 and eval_result.status != _EvalStatus.EVALUATED:
        tf.compat.v1.logging.warning('EvalSpec.throttle_secs is set as 0. This might overload the job before finding (next) new checkpoint. Please consider to increase it.')
    return (eval_result, should_early_stop)