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
def _test_evaluation_for_simple_data(self, n_classes):
    label = 1
    age = 1.0
    age_weight = [[-11.0]] if n_classes == 2 else np.reshape(-11.0 * np.array(list(range(n_classes)), dtype=np.float32), (1, n_classes))
    bias = [-30.0] if n_classes == 2 else [-30.0] * n_classes
    with tf.Graph().as_default():
        tf.Variable(age_weight, name=AGE_WEIGHT_NAME)
        tf.Variable(bias, name=BIAS_NAME)
        tf.Variable(100, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
        save_variables_to_ckpt(self._model_dir)
    est = self._linear_classifier_fn(feature_columns=(self._fc_lib.numeric_column('age'),), n_classes=n_classes, model_dir=self._model_dir)
    eval_metrics = est.evaluate(input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=1)
    if n_classes == 2:
        expected_metrics = {metric_keys.MetricKeys.LOSS: 41.0, tf.compat.v1.GraphKeys.GLOBAL_STEP: 100, metric_keys.MetricKeys.LOSS_MEAN: 41.0, metric_keys.MetricKeys.ACCURACY: 0.0, metric_keys.MetricKeys.PRECISION: 0.0, metric_keys.MetricKeys.RECALL: 0.0, metric_keys.MetricKeys.PREDICTION_MEAN: 0.0, metric_keys.MetricKeys.LABEL_MEAN: 1.0, metric_keys.MetricKeys.ACCURACY_BASELINE: 1, metric_keys.MetricKeys.AUC: 0.0, metric_keys.MetricKeys.AUC_PR: 1.0}
    else:
        logits = age_weight * age + bias
        logits_exp = np.exp(logits)
        softmax = logits_exp / logits_exp.sum()
        expected_loss = -1 * math.log(softmax[0, label])
        expected_metrics = {metric_keys.MetricKeys.LOSS: expected_loss, metric_keys.MetricKeys.LOSS_MEAN: expected_loss, tf.compat.v1.GraphKeys.GLOBAL_STEP: 100, metric_keys.MetricKeys.ACCURACY: 0.0}
    self.assertAllClose(sorted_key_dict(expected_metrics), sorted_key_dict(eval_metrics), rtol=0.001)