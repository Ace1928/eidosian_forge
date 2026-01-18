from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import dnn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _train_op_fn(loss):
    """Returns the op to optimize the loss."""
    train_ops = []
    global_step = tf.compat.v1.train.get_global_step()
    if dnn_logits is not None:
        train_ops.append(dnn_optimizer.minimize(loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=dnn_absolute_scope)))
    if linear_logits is not None:
        train_ops.append(linear_optimizer.minimize(loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=linear_absolute_scope)))
    train_op = tf.group(*train_ops)
    with tf.control_dependencies([train_op]):
        return tf.compat.v1.assign_add(global_step, 1).op