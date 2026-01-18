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
class BaseLinearRegressorIntegrationTest(object):

    def __init__(self, linear_regressor_fn, fc_lib=feature_column):
        self._linear_regressor_fn = linear_regressor_fn
        self._fc_lib = fc_lib

    def setUp(self):
        self._model_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self._model_dir:
            tf.compat.v1.summary.FileWriterCache.clear()
            shutil.rmtree(self._model_dir)

    def _test_complete_flow(self, train_input_fn, eval_input_fn, predict_input_fn, input_dimension, label_dimension, prediction_length):
        feature_columns = [self._fc_lib.numeric_column('x', shape=(input_dimension,))]
        est = self._linear_regressor_fn(feature_columns=feature_columns, label_dimension=label_dimension, model_dir=self._model_dir)
        est.train(train_input_fn, steps=200)
        scores = est.evaluate(eval_input_fn)
        self.assertEqual(200, scores[tf.compat.v1.GraphKeys.GLOBAL_STEP])
        self.assertIn(metric_keys.MetricKeys.LOSS, six.iterkeys(scores))
        predictions = np.array([x['predictions'] for x in est.predict(predict_input_fn)])
        self.assertAllEqual((prediction_length, label_dimension), predictions.shape)
        feature_spec = tf.compat.v1.feature_column.make_parse_example_spec(feature_columns)
        serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(feature_spec)
        export_dir = est.export_saved_model(tempfile.mkdtemp(), serving_input_receiver_fn)
        self.assertTrue(tf.compat.v1.gfile.Exists(export_dir))

    def test_numpy_input_fn(self):
        """Tests complete flow with numpy_input_fn."""
        label_dimension = 2
        input_dimension = label_dimension
        batch_size = 10
        prediction_length = batch_size
        data = np.linspace(0.0, 2.0, batch_size * label_dimension, dtype=np.float32)
        data = data.reshape(batch_size, label_dimension)
        train_input_fn = numpy_io.numpy_input_fn(x={'x': data}, y=data, batch_size=batch_size, num_epochs=None, shuffle=True)
        eval_input_fn = numpy_io.numpy_input_fn(x={'x': data}, y=data, batch_size=batch_size, num_epochs=1, shuffle=False)
        predict_input_fn = numpy_io.numpy_input_fn(x={'x': data}, y=None, batch_size=batch_size, num_epochs=1, shuffle=False)
        self._test_complete_flow(train_input_fn=train_input_fn, eval_input_fn=eval_input_fn, predict_input_fn=predict_input_fn, input_dimension=input_dimension, label_dimension=label_dimension, prediction_length=prediction_length)

    def test_pandas_input_fn(self):
        """Tests complete flow with pandas_input_fn."""
        if not HAS_PANDAS:
            return
        label_dimension = 1
        input_dimension = label_dimension
        batch_size = 10
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        x = pd.DataFrame({'x': data})
        y = pd.Series(data)
        prediction_length = 4
        train_input_fn = pandas_io.pandas_input_fn(x=x, y=y, batch_size=batch_size, num_epochs=None, shuffle=True)
        eval_input_fn = pandas_io.pandas_input_fn(x=x, y=y, batch_size=batch_size, shuffle=False)
        predict_input_fn = pandas_io.pandas_input_fn(x=x, batch_size=batch_size, shuffle=False)
        self._test_complete_flow(train_input_fn=train_input_fn, eval_input_fn=eval_input_fn, predict_input_fn=predict_input_fn, input_dimension=input_dimension, label_dimension=label_dimension, prediction_length=prediction_length)

    def test_input_fn_from_parse_example(self):
        """Tests complete flow with input_fn constructed from parse_example."""
        label_dimension = 2
        input_dimension = label_dimension
        batch_size = 10
        prediction_length = batch_size
        data = np.linspace(0.0, 2.0, batch_size * label_dimension, dtype=np.float32)
        data = data.reshape(batch_size, label_dimension)
        serialized_examples = []
        for datum in data:
            example = example_pb2.Example(features=feature_pb2.Features(feature={'x': feature_pb2.Feature(float_list=feature_pb2.FloatList(value=datum)), 'y': feature_pb2.Feature(float_list=feature_pb2.FloatList(value=datum[:label_dimension]))}))
            serialized_examples.append(example.SerializeToString())
        feature_spec = {'x': tf.io.FixedLenFeature([input_dimension], tf.dtypes.float32), 'y': tf.io.FixedLenFeature([label_dimension], tf.dtypes.float32)}

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
        self._test_complete_flow(train_input_fn=_train_input_fn, eval_input_fn=_eval_input_fn, predict_input_fn=_predict_input_fn, input_dimension=input_dimension, label_dimension=label_dimension, prediction_length=prediction_length)