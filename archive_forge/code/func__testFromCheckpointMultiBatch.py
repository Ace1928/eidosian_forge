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
def _testFromCheckpointMultiBatch(self, n_classes):
    label = [1, 0]
    age = [17.0, 18.5]
    age_weight = [[2.0]] if n_classes == 2 else np.reshape(2.0 * np.array(list(range(n_classes)), dtype=np.float32), (1, n_classes))
    bias = [-35.0] if n_classes == 2 else [-35.0] * n_classes
    initial_global_step = 100
    with tf.Graph().as_default():
        tf.Variable(age_weight, name=AGE_WEIGHT_NAME)
        tf.Variable(bias, name=BIAS_NAME)
        tf.Variable(initial_global_step, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
        save_variables_to_ckpt(self._model_dir)
    if n_classes == 2:
        expected_loss = 1.3133 + 2.1269
    else:
        logits = age_weight * np.reshape(age, (2, 1)) + bias
        logits_exp = np.exp(logits)
        softmax_row_0 = logits_exp[0] / logits_exp[0].sum()
        softmax_row_1 = logits_exp[1] / logits_exp[1].sum()
        expected_loss_0 = -1 * math.log(softmax_row_0[label[0]])
        expected_loss_1 = -1 * math.log(softmax_row_1[label[1]])
        expected_loss = expected_loss_0 + expected_loss_1
    mock_optimizer = self._mock_optimizer(expected_loss=expected_loss)
    est = linear.LinearClassifier(feature_columns=(self._fc_lib.numeric_column('age'),), n_classes=n_classes, optimizer=mock_optimizer, model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)
    num_steps = 10
    est.train(input_fn=lambda: ({'age': age}, label), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(n_classes, expected_global_step=initial_global_step + num_steps, expected_age_weight=age_weight, expected_bias=bias)