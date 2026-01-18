import collections
import operator
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
class _CheckForStoppingHook(tf.compat.v1.train.SessionRunHook):
    """Hook that requests stop if stop is requested by `_StopOnPredicateHook`."""

    def __init__(self):
        self._stop_var = None

    def begin(self):
        self._stop_var = _get_or_create_stop_var()

    def before_run(self, run_context):
        del run_context
        return tf.compat.v1.train.SessionRunArgs(self._stop_var)

    def after_run(self, run_context, run_values):
        should_early_stop = run_values.results
        if should_early_stop:
            tf.compat.v1.logging.info('Early stopping requested, suspending run.')
            run_context.request_stop()