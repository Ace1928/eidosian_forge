from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.inputs import pandas_io
def _mock_optimizer(self, expected_loss=None):
    expected_var_names = ['%s/part_0:0' % AGE_WEIGHT_NAME, '%s/part_0:0' % BIAS_NAME]

    def _minimize(loss, global_step):
        trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        self.assertItemsEqual(expected_var_names, [var.name for var in trainable_vars])
        self.assertEquals(0, loss.shape.ndims)
        if expected_loss is None:
            return tf.compat.v1.assign_add(global_step, 1).op
        assert_loss = assert_close(tf.cast(expected_loss, name='expected', dtype=tf.dtypes.float32), loss, name='assert_loss')
        with tf.control_dependencies((assert_loss,)):
            return tf.compat.v1.assign_add(global_step, 1).op
    mock_optimizer = tf.compat.v1.test.mock.NonCallableMock(spec=tf.compat.v1.train.Optimizer, wraps=tf.compat.v1.train.Optimizer(use_locking=False, name='my_optimizer'))
    mock_optimizer.minimize = tf.compat.v1.test.mock.MagicMock(wraps=_minimize)
    mock_optimizer.__deepcopy__ = lambda _: mock_optimizer
    return mock_optimizer