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
def _test_input_fn_from_parse_example(self, n_classes):
    """Tests complete flow with input_fn constructed from parse_example."""
    input_dimension = 2
    batch_size = 10
    prediction_length = batch_size
    data = np.linspace(0.0, 2.0, batch_size * input_dimension, dtype=np.float32)
    data = data.reshape(batch_size, input_dimension)
    target = np.array([1] * batch_size, dtype=np.int64)
    serialized_examples = []
    for x, y in zip(data, target):
        example = example_pb2.Example(features=feature_pb2.Features(feature={'x': feature_pb2.Feature(float_list=feature_pb2.FloatList(value=x)), 'y': feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[y]))}))
        serialized_examples.append(example.SerializeToString())
    feature_spec = {'x': tf.io.FixedLenFeature([input_dimension], tf.dtypes.float32), 'y': tf.io.FixedLenFeature([1], tf.dtypes.int64)}

    def _train_input_fn():
        feature_map = tf.compat.v1.io.parse_example(serialized_examples, feature_spec)
        features = queue_parsed_features(feature_map)
        labels = features.pop('y')
        return (features, labels)

    def _eval_input_fn():
        feature_map = tf.compat.v1.io.parse_example(tf.compat.v1.train.limit_epochs(serialized_examples, num_epochs=1), feature_spec)
        features = queue_parsed_features(feature_map)
        labels = features.pop('y')
        return (features, labels)

    def _predict_input_fn():
        feature_map = tf.compat.v1.io.parse_example(tf.compat.v1.train.limit_epochs(serialized_examples, num_epochs=1), feature_spec)
        features = queue_parsed_features(feature_map)
        features.pop('y')
        return (features, None)
    self._test_complete_flow(n_classes=n_classes, train_input_fn=_train_input_fn, eval_input_fn=_eval_input_fn, predict_input_fn=_predict_input_fn, input_dimension=input_dimension, prediction_length=prediction_length)