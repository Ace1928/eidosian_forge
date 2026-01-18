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
def _check_no_sync_replicas_optimizer(optimizer):
    if isinstance(optimizer, tf.compat.v1.train.SyncReplicasOptimizer):
        raise ValueError('SyncReplicasOptimizer does not support multi optimizers case. Therefore, it is not supported in DNNLinearCombined model. If you want to use this optimizer, please use either DNN or Linear model.')