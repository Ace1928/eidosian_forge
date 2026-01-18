import collections
import operator
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
class _MultiWorkerEarlyStoppingHook(tf.compat.v1.train.SessionRunHook):
    """Hook that requests stop when `should_stop_fn` returns `True`."""

    def _get_or_create_stop_var_with_aggregation(self):
        with tf.compat.v1.variable_scope(name_or_scope='signal_early_stopping', values=[], reuse=tf.compat.v1.AUTO_REUSE):
            return tf.compat.v1.get_variable(name='STOP', shape=[], dtype=tf.dtypes.int32, initializer=tf.compat.v1.keras.initializers.constant(0), collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], synchronization=tf.VariableSynchronization.ON_WRITE, aggregation=tf.compat.v1.VariableAggregation.SUM, trainable=False)

    def __init__(self, should_stop_fn, run_every_steps=None):
        if not callable(should_stop_fn):
            raise TypeError('`should_stop_fn` must be callable.')
        self._should_stop_fn = should_stop_fn
        self._timer = tf.compat.v1.train.SecondOrStepTimer(every_secs=None, every_steps=run_every_steps)
        self._global_step_tensor = None
        self._stop_var = None
        self._stop_op = None
        self._non_stop_op = None

    def begin(self):
        self._global_step_tensor = tf.compat.v1.train.get_global_step()
        self._stop_var = self._get_or_create_stop_var_with_aggregation()
        assert tf.distribute.in_cross_replica_context()
        strategy = tf.distribute.get_strategy()
        self._stop_placeholder = None

        def stop_op_fn(var):
            placeholder = tf.compat.v1.placeholder_with_default(0, tuple(), name='stop_value')
            if self._stop_placeholder is None:
                self._stop_placeholder = placeholder
            return var.assign_add(placeholder)
        self._stop_op = strategy.run(stop_op_fn, args=(self._stop_var,))

    def before_run(self, run_context):
        del run_context
        return tf.compat.v1.train.SessionRunArgs({'global_step': self._global_step_tensor, 'stop_var': self._stop_var})

    def after_run(self, run_context, run_values):
        global_step = run_values.results['global_step']
        should_early_stop = run_values.results['stop_var']
        if should_early_stop > 0:
            tf.compat.v1.logging.info('Early stopping requested, suspending run.')
            run_context.request_stop()
            return
        if self._timer.should_trigger_for_step(global_step):
            self._timer.update_last_triggered_step(global_step)
            if self._should_stop_fn():
                run_context.session.run(self._stop_op, feed_dict={self._stop_placeholder: 1})
                tf.compat.v1.logging.info('Requesting early stopping at global step %d', global_step)
            else:
                run_context.session.run(self._stop_op, feed_dict={self._stop_placeholder: 0})