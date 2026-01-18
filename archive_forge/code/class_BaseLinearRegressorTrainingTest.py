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
class BaseLinearRegressorTrainingTest(object):

    def __init__(self, linear_regressor_fn, fc_lib=feature_column):
        self._linear_regressor_fn = linear_regressor_fn
        self._fc_lib = fc_lib

    def setUp(self):
        self._model_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self._model_dir:
            tf.compat.v1.summary.FileWriterCache.clear()
            shutil.rmtree(self._model_dir)

    def _mock_optimizer(self, expected_loss=None):
        expected_var_names = ['%s/part_0:0' % AGE_WEIGHT_NAME, '%s/part_0:0' % BIAS_NAME]

        def _minimize(loss, global_step=None, var_list=None):
            trainable_vars = var_list or tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            self.assertItemsEqual(expected_var_names, [var.name for var in trainable_vars])
            self.assertEquals(0, loss.shape.ndims)
            if expected_loss is None:
                if global_step is not None:
                    return tf.compat.v1.assign_add(global_step, 1).op
                return tf.no_op()
            assert_loss = assert_close(tf.cast(expected_loss, name='expected', dtype=tf.dtypes.float32), loss, name='assert_loss')
            with tf.control_dependencies((assert_loss,)):
                if global_step is not None:
                    return tf.compat.v1.assign_add(global_step, 1).op
                return tf.no_op()
        mock_optimizer = tf.compat.v1.test.mock.NonCallableMock(spec=tf.compat.v1.train.Optimizer, wraps=tf.compat.v1.train.Optimizer(use_locking=False, name='my_optimizer'))
        mock_optimizer.minimize = tf.compat.v1.test.mock.MagicMock(wraps=_minimize)
        mock_optimizer.__deepcopy__ = lambda _: mock_optimizer
        return mock_optimizer

    def _assert_checkpoint(self, expected_global_step, expected_age_weight=None, expected_bias=None):
        shapes = {name: shape for name, shape in tf.train.list_variables(self._model_dir)}
        self.assertEqual([], shapes[tf.compat.v1.GraphKeys.GLOBAL_STEP])
        self.assertEqual(expected_global_step, tf.train.load_variable(self._model_dir, tf.compat.v1.GraphKeys.GLOBAL_STEP))
        self.assertEqual([1, 1], shapes[AGE_WEIGHT_NAME])
        if expected_age_weight is not None:
            self.assertEqual(expected_age_weight, tf.train.load_variable(self._model_dir, AGE_WEIGHT_NAME))
        self.assertEqual([1], shapes[BIAS_NAME])
        if expected_bias is not None:
            self.assertEqual(expected_bias, tf.train.load_variable(self._model_dir, BIAS_NAME))

    def testFromScratchWithDefaultOptimizer(self):
        label = 5.0
        age = 17
        linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('age'),), model_dir=self._model_dir)
        num_steps = 10
        linear_regressor.train(input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
        self._assert_checkpoint(num_steps)

    def testTrainWithOneDimLabel(self):
        label_dimension = 1
        batch_size = 20
        feature_columns = [self._fc_lib.numeric_column('age', shape=(1,))]
        est = self._linear_regressor_fn(feature_columns=feature_columns, label_dimension=label_dimension, model_dir=self._model_dir)
        data_rank_1 = np.linspace(0.0, 2.0, batch_size, dtype=np.float32)
        self.assertEqual((batch_size,), data_rank_1.shape)
        train_input_fn = numpy_io.numpy_input_fn(x={'age': data_rank_1}, y=data_rank_1, batch_size=batch_size, num_epochs=None, shuffle=True)
        est.train(train_input_fn, steps=200)
        self._assert_checkpoint(200)

    def testTrainWithOneDimWeight(self):
        label_dimension = 1
        batch_size = 20
        feature_columns = [self._fc_lib.numeric_column('age', shape=(1,))]
        est = self._linear_regressor_fn(feature_columns=feature_columns, label_dimension=label_dimension, weight_column='w', model_dir=self._model_dir)
        data_rank_1 = np.linspace(0.0, 2.0, batch_size, dtype=np.float32)
        self.assertEqual((batch_size,), data_rank_1.shape)
        train_input_fn = numpy_io.numpy_input_fn(x={'age': data_rank_1, 'w': data_rank_1}, y=data_rank_1, batch_size=batch_size, num_epochs=None, shuffle=True)
        est.train(train_input_fn, steps=200)
        self._assert_checkpoint(200)

    def testFromScratch(self):
        label = 5.0
        age = 17
        mock_optimizer = self._mock_optimizer(expected_loss=25.0)
        linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('age'),), model_dir=self._model_dir, optimizer=mock_optimizer)
        self.assertEqual(0, mock_optimizer.minimize.call_count)
        num_steps = 10
        linear_regressor.train(input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
        self.assertEqual(1, mock_optimizer.minimize.call_count)
        self._assert_checkpoint(expected_global_step=num_steps, expected_age_weight=0.0, expected_bias=0.0)

    def testFromCheckpoint(self):
        age_weight = 10.0
        bias = 5.0
        initial_global_step = 100
        with tf.Graph().as_default():
            tf.Variable([[age_weight]], name=AGE_WEIGHT_NAME)
            tf.Variable([bias], name=BIAS_NAME)
            tf.Variable(initial_global_step, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        mock_optimizer = self._mock_optimizer(expected_loss=28900.0)
        linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('age'),), model_dir=self._model_dir, optimizer=mock_optimizer)
        self.assertEqual(0, mock_optimizer.minimize.call_count)
        num_steps = 10
        linear_regressor.train(input_fn=lambda: ({'age': ((17,),)}, ((5.0,),)), steps=num_steps)
        self.assertEqual(1, mock_optimizer.minimize.call_count)
        self._assert_checkpoint(expected_global_step=initial_global_step + num_steps, expected_age_weight=age_weight, expected_bias=bias)

    def testFromCheckpointMultiBatch(self):
        age_weight = 10.0
        bias = 5.0
        initial_global_step = 100
        with tf.Graph().as_default():
            tf.Variable([[age_weight]], name=AGE_WEIGHT_NAME)
            tf.Variable([bias], name=BIAS_NAME)
            tf.Variable(initial_global_step, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        mock_optimizer = self._mock_optimizer(expected_loss=52004.0)
        linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('age'),), model_dir=self._model_dir, optimizer=mock_optimizer)
        self.assertEqual(0, mock_optimizer.minimize.call_count)
        num_steps = 10
        linear_regressor.train(input_fn=lambda: ({'age': ((17,), (15,))}, ((5.0,), (3.0,))), steps=num_steps)
        self.assertEqual(1, mock_optimizer.minimize.call_count)
        self._assert_checkpoint(expected_global_step=initial_global_step + num_steps, expected_age_weight=age_weight, expected_bias=bias)