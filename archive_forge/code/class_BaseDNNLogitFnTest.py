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
class BaseDNNLogitFnTest(object):
    """Tests correctness of logits calculated from _dnn_logit_fn_builder."""

    def __init__(self, dnn_logit_fn_builder, fc_impl=feature_column):
        self._dnn_logit_fn_builder = dnn_logit_fn_builder
        self._fc_impl = fc_impl

    def setUp(self):
        self._model_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self._model_dir:
            tf.compat.v1.summary.FileWriterCache.clear()
            shutil.rmtree(self._model_dir)

    def _test_logits(self, mode, hidden_units, logits_dimension, inputs, expected_logits, batch_norm=False):
        """Tests that the expected logits are calculated."""
        with tf.Graph().as_default():
            tf.compat.v1.train.create_global_step()
            with tf.compat.v1.variable_scope('dnn'):
                input_layer_partitioner = tf.compat.v1.min_max_variable_partitioner(max_partitions=0, min_slice_size=64 << 20)
                logit_fn = self._dnn_logit_fn_builder(units=logits_dimension, hidden_units=hidden_units, feature_columns=[self._fc_impl.numeric_column('age', shape=np.array(inputs).shape[1:])], activation_fn=tf.nn.relu, dropout=None, input_layer_partitioner=input_layer_partitioner, batch_norm=batch_norm)
                logits = logit_fn(features={'age': tf.constant(inputs)}, mode=mode)
                with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=self._model_dir) as sess:
                    self.assertAllClose(expected_logits, sess.run(logits))

    def test_one_dim_logits(self):
        """Tests one-dimensional logits.

    input_layer = [[10]]
    hidden_layer_0 = [[relu(0.6*10 +0.1), relu(0.5*10 -0.1)]] = [[6.1, 4.9]]
    hidden_layer_1 = [[relu(1*6.1 -0.8*4.9 +0.2), relu(0.8*6.1 -1*4.9 -0.1)]]
                   = [[relu(2.38), relu(-0.12)]] = [[2.38, 0]]
    logits = [[-1*2.38 +1*0 +0.3]] = [[-2.08]]
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0], [1.0]], [0.3])), base_global_step, self._model_dir)
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            self._test_logits(mode, hidden_units=(2, 2), logits_dimension=1, inputs=[[10.0]], expected_logits=[[-2.08]])

    def test_one_dim_logits_with_batch_norm(self):
        """Tests one-dimensional logits.

    input_layer = [[10]]
    hidden_layer_0 = [[relu(0.6*10 +1), relu(0.5*10 -1)]] = [[7, 4]]
    hidden_layer_0 = [[relu(0.6*20 +1), relu(0.5*20 -1)]] = [[13, 9]]

    batch_norm_0, training (epsilon = 0.001):
      mean1 = 1/2*(7+13) = 10,
      variance1 = 1/2*(3^2+3^2) = 9
      x11 = (7-10)/sqrt(9+0.001) = -0.999944449,
      x21 = (13-10)/sqrt(9+0.001) = 0.999944449,

      mean2 = 1/2*(4+9) = 6.5,
      variance2 = 1/2*(2.5^2+.2.5^2) = 6.25
      x12 = (4-6.5)/sqrt(6.25+0.001) = -0.99992001,
      x22 = (9-6.5)/sqrt(6.25+0.001) = 0.99992001,

    logits = [[-1*(-0.999944449) + 2*(-0.99992001) + 0.3],
              [-1*0.999944449 + 2*0.99992001 + 0.3]]
           = [[-0.699895571],[1.299895571]]

    batch_norm_0, not training (epsilon = 0.001):
      moving_mean1 = 0, moving_variance1 = 1
      x11 = (7-0)/sqrt(1+0.001) = 6.996502623,
      x21 = (13-0)/sqrt(1+0.001) = 12.993504871,
      moving_mean2 = 0, moving_variance2 = 1
      x12 = (4-0)/sqrt(1+0.001) = 3.998001499,
      x22 = (9-0)/sqrt(1+0.001) = 8.995503372,

    logits = [[-1*6.996502623 + 2*3.998001499 + 0.3],
              [-1*12.993504871 + 2*8.995503372 + 0.3]]
           = [[1.299500375],[5.297501873]]
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5]], [1.0, -1.0]), ([[-1.0], [2.0]], [0.3])), base_global_step, self._model_dir, batch_norm_vars=([[0, 0], [1, 1], [0, 0], [1, 1]],))
        self._test_logits(ModeKeys.TRAIN, hidden_units=[2], logits_dimension=1, inputs=[[10.0], [20.0]], expected_logits=[[-0.699895571], [1.299895571]], batch_norm=True)
        for mode in [ModeKeys.EVAL, ModeKeys.PREDICT]:
            self._test_logits(mode, hidden_units=[2], logits_dimension=1, inputs=[[10.0], [20.0]], expected_logits=[[1.299500375], [5.297501873]], batch_norm=True)

    def test_multi_dim_logits(self):
        """Tests multi-dimensional logits.

    input_layer = [[10]]
    hidden_layer_0 = [[relu(0.6*10 +0.1), relu(0.5*10 -0.1)]] = [[6.1, 4.9]]
    hidden_layer_1 = [[relu(1*6.1 -0.8*4.9 +0.2), relu(0.8*6.1 -1*4.9 -0.1)]]
                   = [[relu(2.38), relu(-0.12)]] = [[2.38, 0]]
    logits = [[-1*2.38 +0.3, 1*2.38 -0.3, 0.5*2.38]]
           = [[-2.08, 2.08, 1.19]]
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), base_global_step, self._model_dir)
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            self._test_logits(mode, hidden_units=(2, 2), logits_dimension=3, inputs=[[10.0]], expected_logits=[[-2.08, 2.08, 1.19]])

    def test_multi_example_multi_dim_logits(self):
        """Tests multiple examples and multi-dimensional logits.

    input_layer = [[10], [5]]
    hidden_layer_0 = [[relu(0.6*10 +0.1), relu(0.5*10 -0.1)],
                      [relu(0.6*5 +0.1), relu(0.5*5 -0.1)]]
                   = [[6.1, 4.9], [3.1, 2.4]]
    hidden_layer_1 = [[relu(1*6.1 -0.8*4.9 +0.2), relu(0.8*6.1 -1*4.9 -0.1)],
                      [relu(1*3.1 -0.8*2.4 +0.2), relu(0.8*3.1 -1*2.4 -0.1)]]
                   = [[2.38, 0], [1.38, 0]]
    logits = [[-1*2.38 +0.3, 1*2.38 -0.3, 0.5*2.38],
              [-1*1.38 +0.3, 1*1.38 -0.3, 0.5*1.38]]
           = [[-2.08, 2.08, 1.19], [-1.08, 1.08, 0.69]]
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), base_global_step, self._model_dir)
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            self._test_logits(mode, hidden_units=(2, 2), logits_dimension=3, inputs=[[10.0], [5.0]], expected_logits=[[-2.08, 2.08, 1.19], [-1.08, 1.08, 0.69]])

    def test_multi_dim_input_one_dim_logits(self):
        """Tests multi-dimensional inputs and one-dimensional logits.

    input_layer = [[10, 8]]
    hidden_layer_0 = [[relu(0.6*10 -0.6*8 +0.1), relu(0.5*10 -0.5*8 -0.1)]]
                   = [[1.3, 0.9]]
    hidden_layer_1 = [[relu(1*1.3 -0.8*0.9 + 0.2), relu(0.8*1.3 -1*0.9 -0.2)]]
                   = [[0.78, relu(-0.06)]] = [[0.78, 0]]
    logits = [[-1*0.78 +1*0 +0.3]] = [[-0.48]]
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0], [1.0]], [0.3])), base_global_step, self._model_dir)
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            self._test_logits(mode, hidden_units=(2, 2), logits_dimension=1, inputs=[[10.0, 8.0]], expected_logits=[[-0.48]])

    def test_multi_dim_input_multi_dim_logits(self):
        """Tests multi-dimensional inputs and multi-dimensional logits.

    input_layer = [[10, 8]]
    hidden_layer_0 = [[relu(0.6*10 -0.6*8 +0.1), relu(0.5*10 -0.5*8 -0.1)]]
                   = [[1.3, 0.9]]
    hidden_layer_1 = [[relu(1*1.3 -0.8*0.9 + 0.2), relu(0.8*1.3 -1*0.9 -0.2)]]
                   = [[0.78, relu(-0.06)]] = [[0.78, 0]]
    logits = [[-1*0.78 + 0.3, 1*0.78 -0.3, 0.5*0.78]] = [[-0.48, 0.48, 0.39]]
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), base_global_step, self._model_dir)
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            self._test_logits(mode, hidden_units=(2, 2), logits_dimension=3, inputs=[[10.0, 8.0]], expected_logits=[[-0.48, 0.48, 0.39]])

    def test_multi_feature_column_multi_dim_logits(self):
        """Tests multiple feature columns and multi-dimensional logits.

    All numbers are the same as test_multi_dim_input_multi_dim_logits. The only
    difference is that the input consists of two 1D feature columns, instead of
    one 2D feature column.
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), base_global_step, self._model_dir)
        hidden_units = (2, 2)
        logits_dimension = 3
        inputs = ([[10.0]], [[8.0]])
        expected_logits = [[-0.48, 0.48, 0.39]]
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            with tf.Graph().as_default():
                tf.compat.v1.train.create_global_step()
                with tf.compat.v1.variable_scope('dnn'):
                    input_layer_partitioner = tf.compat.v1.min_max_variable_partitioner(max_partitions=0, min_slice_size=64 << 20)
                    logit_fn = self._dnn_logit_fn_builder(units=logits_dimension, hidden_units=hidden_units, feature_columns=[self._fc_impl.numeric_column('age'), self._fc_impl.numeric_column('height')], activation_fn=tf.nn.relu, dropout=None, input_layer_partitioner=input_layer_partitioner, batch_norm=False)
                    logits = logit_fn(features={'age': tf.constant(inputs[0]), 'height': tf.constant(inputs[1])}, mode=mode)
                    with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=self._model_dir) as sess:
                        self.assertAllClose(expected_logits, sess.run(logits))

    def test_multi_feature_column_mix_multi_dim_logits(self):
        """Tests multiple feature columns and multi-dimensional logits.

    All numbers are the same as test_multi_dim_input_multi_dim_logits. The only
    difference is that the input consists of two 1D feature columns, instead of
    one 2D feature column.
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), base_global_step, self._model_dir)
        hidden_units = (2, 2)
        logits_dimension = 3
        inputs = ([[10.0]], [[8.0]])
        expected_logits = [[-0.48, 0.48, 0.39]]
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            with tf.Graph().as_default():
                tf.compat.v1.train.create_global_step()
                with tf.compat.v1.variable_scope('dnn'):
                    input_layer_partitioner = tf.compat.v1.min_max_variable_partitioner(max_partitions=0, min_slice_size=64 << 20)
                    logit_fn = self._dnn_logit_fn_builder(units=logits_dimension, hidden_units=hidden_units, feature_columns=[feature_column.numeric_column('age'), tf.feature_column.numeric_column('height')], activation_fn=tf.nn.relu, dropout=None, input_layer_partitioner=input_layer_partitioner, batch_norm=False)
                    logits = logit_fn(features={'age': tf.constant(inputs[0]), 'height': tf.constant(inputs[1])}, mode=mode)
                    with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=self._model_dir) as sess:
                        self.assertAllClose(expected_logits, sess.run(logits))