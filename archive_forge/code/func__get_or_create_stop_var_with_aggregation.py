import collections
import operator
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _get_or_create_stop_var_with_aggregation(self):
    with tf.compat.v1.variable_scope(name_or_scope='signal_early_stopping', values=[], reuse=tf.compat.v1.AUTO_REUSE):
        return tf.compat.v1.get_variable(name='STOP', shape=[], dtype=tf.dtypes.int32, initializer=tf.compat.v1.keras.initializers.constant(0), collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], synchronization=tf.VariableSynchronization.ON_WRITE, aggregation=tf.compat.v1.VariableAggregation.SUM, trainable=False)