import collections
import operator
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
class _StopOnPredicateHook(tf.compat.v1.train.SessionRunHook):
    """Hook that requests stop when `should_stop_fn` returns `True`."""

    def __init__(self, should_stop_fn, run_every_secs=60, run_every_steps=None):
        if not callable(should_stop_fn):
            raise TypeError('`should_stop_fn` must be callable.')
        self._should_stop_fn = should_stop_fn
        self._timer = tf.compat.v1.train.SecondOrStepTimer(every_secs=run_every_secs, every_steps=run_every_steps)
        self._global_step_tensor = None
        self._stop_var = None
        self._stop_op = None

    def begin(self):
        self._global_step_tensor = tf.compat.v1.train.get_global_step()
        self._stop_var = _get_or_create_stop_var()
        self._stop_op = tf.compat.v1.assign(self._stop_var, True)

    def before_run(self, run_context):
        del run_context
        return tf.compat.v1.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            self._timer.update_last_triggered_step(global_step)
            if self._should_stop_fn():
                tf.compat.v1.logging.info('Requesting early stopping at global step %d', global_step)
                run_context.session.run(self._stop_op)
                run_context.request_stop()