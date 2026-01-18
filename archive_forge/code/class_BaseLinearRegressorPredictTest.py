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
class BaseLinearRegressorPredictTest(object):

    def __init__(self, linear_regressor_fn, fc_lib=feature_column):
        self._linear_regressor_fn = linear_regressor_fn
        self._fc_lib = fc_lib

    def setUp(self):
        self._model_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self._model_dir:
            tf.compat.v1.summary.FileWriterCache.clear()
            shutil.rmtree(self._model_dir)

    def test_1d(self):
        """Tests predict when all variables are one-dimensional."""
        with tf.Graph().as_default():
            tf.Variable([[10.0]], name='linear/linear_model/x/weights')
            tf.Variable([0.2], name=BIAS_NAME)
            tf.Variable(100, name='global_step', dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('x'),), model_dir=self._model_dir)
        predict_input_fn = numpy_io.numpy_input_fn(x={'x': np.array([[2.0]])}, y=None, batch_size=1, num_epochs=1, shuffle=False)
        predictions = linear_regressor.predict(input_fn=predict_input_fn)
        predicted_scores = list([x['predictions'] for x in predictions])
        self.assertAllClose([[20.2]], predicted_scores)

    def testMultiDim(self):
        """Tests predict when all variables are multi-dimenstional."""
        batch_size = 2
        label_dimension = 3
        x_dim = 4
        feature_columns = (self._fc_lib.numeric_column('x', shape=(x_dim,)),)
        with tf.Graph().as_default():
            tf.Variable([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]], name='linear/linear_model/x/weights')
            tf.Variable([0.2, 0.4, 0.6], name=BIAS_NAME)
            tf.Variable(100, name='global_step', dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        linear_regressor = self._linear_regressor_fn(feature_columns=feature_columns, label_dimension=label_dimension, model_dir=self._model_dir)
        predict_input_fn = numpy_io.numpy_input_fn(x={'x': np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])}, y=None, batch_size=batch_size, num_epochs=1, shuffle=False)
        predictions = linear_regressor.predict(input_fn=predict_input_fn)
        predicted_scores = list([x['predictions'] for x in predictions])
        self.assertAllClose([[30.2, 40.4, 50.6], [70.2, 96.4, 122.6]], predicted_scores)

    def testTwoFeatureColumns(self):
        """Tests predict with two feature columns."""
        with tf.Graph().as_default():
            tf.Variable([[10.0]], name='linear/linear_model/x0/weights')
            tf.Variable([[20.0]], name='linear/linear_model/x1/weights')
            tf.Variable([0.2], name=BIAS_NAME)
            tf.Variable(100, name='global_step', dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('x0'), self._fc_lib.numeric_column('x1')), model_dir=self._model_dir)
        predict_input_fn = numpy_io.numpy_input_fn(x={'x0': np.array([[2.0]]), 'x1': np.array([[3.0]])}, y=None, batch_size=1, num_epochs=1, shuffle=False)
        predictions = linear_regressor.predict(input_fn=predict_input_fn)
        predicted_scores = list([x['predictions'] for x in predictions])
        self.assertAllClose([[80.2]], predicted_scores)

    def testTwoFeatureColumnsMix(self):
        """Tests predict with two feature columns."""
        with tf.Graph().as_default():
            tf.Variable([[10.0]], name='linear/linear_model/x0/weights')
            tf.Variable([[20.0]], name='linear/linear_model/x1/weights')
            tf.Variable([0.2], name=BIAS_NAME)
            tf.Variable(100, name='global_step', dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        linear_regressor = self._linear_regressor_fn(feature_columns=(feature_column.numeric_column('x0'), tf.feature_column.numeric_column('x1')), model_dir=self._model_dir)

        def _predict_input_fn():
            return tf.compat.v1.data.Dataset.from_tensor_slices({'x0': np.array([[2.0]]), 'x1': np.array([[3.0]])}).batch(1)
        predictions = linear_regressor.predict(input_fn=_predict_input_fn)
        predicted_scores = list([x['predictions'] for x in predictions])
        self.assertAllClose([[80.2]], predicted_scores)

    def testSparseCombiner(self):
        w_a = 2.0
        w_b = 3.0
        w_c = 5.0
        bias = 5.0
        with tf.Graph().as_default():
            tf.Variable([[w_a], [w_b], [w_c]], name=LANGUAGE_WEIGHT_NAME)
            tf.Variable([bias], name=BIAS_NAME)
            tf.Variable(1, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)

        def _input_fn():
            return tf.compat.v1.data.Dataset.from_tensors({'language': tf.sparse.SparseTensor(values=['a', 'c', 'b', 'c'], indices=[[0, 0], [0, 1], [1, 0], [1, 1]], dense_shape=[2, 2])})
        feature_columns = (self._fc_lib.categorical_column_with_vocabulary_list('language', vocabulary_list=['a', 'b', 'c']),)
        linear_regressor = self._linear_regressor_fn(feature_columns=feature_columns, model_dir=self._model_dir)
        predictions = linear_regressor.predict(input_fn=_input_fn)
        predicted_scores = list([x['predictions'] for x in predictions])
        self.assertAllClose([[12.0], [13.0]], predicted_scores)
        linear_regressor = self._linear_regressor_fn(feature_columns=feature_columns, model_dir=self._model_dir, sparse_combiner='mean')
        predictions = linear_regressor.predict(input_fn=_input_fn)
        predicted_scores = list([x['predictions'] for x in predictions])
        self.assertAllClose([[8.5], [9.0]], predicted_scores)
        linear_regressor = self._linear_regressor_fn(feature_columns=feature_columns, model_dir=self._model_dir, sparse_combiner='sqrtn')
        predictions = linear_regressor.predict(input_fn=_input_fn)
        predicted_scores = list([x['predictions'] for x in predictions])
        self.assertAllClose([[9.94974], [10.65685]], predicted_scores)