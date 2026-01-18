from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.export import export_lib
def _train_ops(self, features):
    """Add training ops to the graph."""
    mode = estimator_lib.ModeKeys.TRAIN
    with tf.compat.v1.variable_scope('model', use_resource=True):
        model_outputs = self.create_loss(features, mode)
    train_op = self.optimizer.minimize(model_outputs.loss, global_step=tf.compat.v1.train.get_global_step())
    return estimator_lib.EstimatorSpec(loss=model_outputs.loss, mode=mode, train_op=train_op)