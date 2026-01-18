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