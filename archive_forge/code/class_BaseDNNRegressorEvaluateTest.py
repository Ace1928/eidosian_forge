from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
class BaseDNNRegressorEvaluateTest(object):

    def __init__(self, dnn_regressor_fn, fc_impl=feature_column):
        self._dnn_regressor_fn = dnn_regressor_fn
        self._fc_impl = fc_impl

    def setUp(self):
        self._model_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self._model_dir:
            tf.compat.v1.summary.FileWriterCache.clear()
            shutil.rmtree(self._model_dir)

    def test_one_dim(self):
        """Asserts evaluation metrics for one-dimensional input and logits."""
        global_step = 100
        create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0], [1.0]], [0.3])), global_step, self._model_dir)
        dnn_regressor = self._dnn_regressor_fn(hidden_units=(2, 2), feature_columns=[self._fc_impl.numeric_column('age')], model_dir=self._model_dir)

        def _input_fn():
            return ({'age': [[10.0]]}, [[1.0]])
        expected_loss = 9.4864
        self.assertAllClose({metric_keys.MetricKeys.LOSS: expected_loss, metric_keys.MetricKeys.LOSS_MEAN: expected_loss, metric_keys.MetricKeys.PREDICTION_MEAN: -2.08, metric_keys.MetricKeys.LABEL_MEAN: 1.0, tf.compat.v1.GraphKeys.GLOBAL_STEP: global_step}, dnn_regressor.evaluate(input_fn=_input_fn, steps=1))

    def test_multi_dim(self):
        """Asserts evaluation metrics for multi-dimensional input and logits."""
        global_step = 100
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), global_step, self._model_dir)
        label_dimension = 3
        dnn_regressor = self._dnn_regressor_fn(hidden_units=(2, 2), feature_columns=[self._fc_impl.numeric_column('age', shape=[2])], label_dimension=label_dimension, model_dir=self._model_dir)

        def _input_fn():
            return ({'age': [[10.0, 8.0]]}, [[1.0, -1.0, 0.5]])
        expected_loss = 4.3929
        self.assertAllClose({metric_keys.MetricKeys.LOSS: expected_loss, metric_keys.MetricKeys.LOSS_MEAN: expected_loss / label_dimension, metric_keys.MetricKeys.PREDICTION_MEAN: 0.39 / 3.0, metric_keys.MetricKeys.LABEL_MEAN: 0.5 / 3.0, tf.compat.v1.GraphKeys.GLOBAL_STEP: global_step}, dnn_regressor.evaluate(input_fn=_input_fn, steps=1))

    def test_multi_dim_weights(self):
        """Asserts evaluation metrics for multi-dimensional input and logits."""
        global_step = 100
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), global_step, self._model_dir)
        label_dimension = 3
        dnn_regressor = self._dnn_regressor_fn(hidden_units=(2, 2), feature_columns=[self._fc_impl.numeric_column('age', shape=[2])], label_dimension=label_dimension, weight_column='w', model_dir=self._model_dir)

        def _input_fn():
            return ({'age': [[10.0, 8.0]], 'w': [10.0]}, [[1.0, -1.0, 0.5]])
        expected_loss = 43.929
        metrics = dnn_regressor.evaluate(input_fn=_input_fn, steps=1)
        self.assertAlmostEqual(expected_loss, metrics[metric_keys.MetricKeys.LOSS], places=3)