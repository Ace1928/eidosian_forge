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
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.inputs import pandas_io
class _Optimizer(tf.keras.optimizers.legacy.Optimizer):

    def get_updates(self, loss, params):
        trainable_vars = params
        testcase.assertItemsEqual(expected_var_names, [var.name for var in trainable_vars])
        testcase.assertEquals(0, loss.shape.ndims)
        if expected_loss is None:
            if self.iterations is not None:
                return [self.iterations.assign_add(1).op]
            return [tf.no_op()]
        assert_loss = assert_close(tf.cast(expected_loss, name='expected', dtype=tf.dtypes.float32), loss, name='assert_loss')
        with tf.control_dependencies((assert_loss,)):
            if self.iterations is not None:
                return [self.iterations.assign_add(1).op]
            return [tf.no_op()]

    def get_config(self):
        config = super(_Optimizer, self).get_config()
        return config