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
class BaseLinearClassifierTrainingTest(object):

    def __init__(self, linear_classifier_fn, fc_lib=feature_column):
        self._linear_classifier_fn = linear_classifier_fn
        self._fc_lib = fc_lib

    def setUp(self):
        self._model_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self._model_dir:
            shutil.rmtree(self._model_dir)

    def _mock_optimizer(self, expected_loss=None):
        expected_var_names = ['%s/part_0:0' % AGE_WEIGHT_NAME, '%s/part_0:0' % BIAS_NAME]

        def _minimize(loss, global_step):
            trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            self.assertItemsEqual(expected_var_names, [var.name for var in trainable_vars])
            self.assertEquals(0, loss.shape.ndims)
            if expected_loss is None:
                return tf.compat.v1.assign_add(global_step, 1).op
            assert_loss = assert_close(tf.cast(expected_loss, name='expected', dtype=tf.dtypes.float32), loss, name='assert_loss')
            with tf.control_dependencies((assert_loss,)):
                return tf.compat.v1.assign_add(global_step, 1).op
        mock_optimizer = tf.compat.v1.test.mock.NonCallableMock(spec=tf.compat.v1.train.Optimizer, wraps=tf.compat.v1.train.Optimizer(use_locking=False, name='my_optimizer'))
        mock_optimizer.minimize = tf.compat.v1.test.mock.MagicMock(wraps=_minimize)
        mock_optimizer.__deepcopy__ = lambda _: mock_optimizer
        return mock_optimizer

    def _assert_checkpoint(self, n_classes, expected_global_step, expected_age_weight=None, expected_bias=None):
        logits_dimension = n_classes if n_classes > 2 else 1
        shapes = {name: shape for name, shape in tf.train.list_variables(self._model_dir)}
        self.assertEqual([], shapes[tf.compat.v1.GraphKeys.GLOBAL_STEP])
        self.assertEqual(expected_global_step, tf.train.load_variable(self._model_dir, tf.compat.v1.GraphKeys.GLOBAL_STEP))
        self.assertEqual([1, logits_dimension], shapes[AGE_WEIGHT_NAME])
        if expected_age_weight is not None:
            self.assertAllEqual(expected_age_weight, tf.train.load_variable(self._model_dir, AGE_WEIGHT_NAME))
        self.assertEqual([logits_dimension], shapes[BIAS_NAME])
        if expected_bias is not None:
            self.assertAllEqual(expected_bias, tf.train.load_variable(self._model_dir, BIAS_NAME))

    def _testFromScratchWithDefaultOptimizer(self, n_classes):
        label = 0
        age = 17
        est = linear.LinearClassifier(feature_columns=(self._fc_lib.numeric_column('age'),), n_classes=n_classes, model_dir=self._model_dir)
        num_steps = 10
        est.train(input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
        self._assert_checkpoint(n_classes, num_steps)

    def testBinaryClassesFromScratchWithDefaultOptimizer(self):
        self._testFromScratchWithDefaultOptimizer(n_classes=2)

    def testMultiClassesFromScratchWithDefaultOptimizer(self):
        self._testFromScratchWithDefaultOptimizer(n_classes=4)

    def _testTrainWithTwoDimsLabel(self, n_classes):
        batch_size = 20
        est = linear.LinearClassifier(feature_columns=(self._fc_lib.numeric_column('age'),), n_classes=n_classes, model_dir=self._model_dir)
        data_rank_1 = np.array([0, 1])
        data_rank_2 = np.array([[0], [1]])
        self.assertEqual((2,), data_rank_1.shape)
        self.assertEqual((2, 1), data_rank_2.shape)
        train_input_fn = numpy_io.numpy_input_fn(x={'age': data_rank_1}, y=data_rank_2, batch_size=batch_size, num_epochs=None, shuffle=True)
        est.train(train_input_fn, steps=200)
        self._assert_checkpoint(n_classes, 200)

    def testBinaryClassesTrainWithTwoDimsLabel(self):
        self._testTrainWithTwoDimsLabel(n_classes=2)

    def testMultiClassesTrainWithTwoDimsLabel(self):
        self._testTrainWithTwoDimsLabel(n_classes=4)

    def _testTrainWithOneDimLabel(self, n_classes):
        batch_size = 20
        est = linear.LinearClassifier(feature_columns=(self._fc_lib.numeric_column('age'),), n_classes=n_classes, model_dir=self._model_dir)
        data_rank_1 = np.array([0, 1])
        self.assertEqual((2,), data_rank_1.shape)
        train_input_fn = numpy_io.numpy_input_fn(x={'age': data_rank_1}, y=data_rank_1, batch_size=batch_size, num_epochs=None, shuffle=True)
        est.train(train_input_fn, steps=200)
        self._assert_checkpoint(n_classes, 200)

    def testBinaryClassesTrainWithOneDimLabel(self):
        self._testTrainWithOneDimLabel(n_classes=2)

    def testMultiClassesTrainWithOneDimLabel(self):
        self._testTrainWithOneDimLabel(n_classes=4)

    def _testTrainWithTwoDimsWeight(self, n_classes):
        batch_size = 20
        est = linear.LinearClassifier(feature_columns=(self._fc_lib.numeric_column('age'),), weight_column='w', n_classes=n_classes, model_dir=self._model_dir)
        data_rank_1 = np.array([0, 1])
        data_rank_2 = np.array([[0], [1]])
        self.assertEqual((2,), data_rank_1.shape)
        self.assertEqual((2, 1), data_rank_2.shape)
        train_input_fn = numpy_io.numpy_input_fn(x={'age': data_rank_1, 'w': data_rank_2}, y=data_rank_1, batch_size=batch_size, num_epochs=None, shuffle=True)
        est.train(train_input_fn, steps=200)
        self._assert_checkpoint(n_classes, 200)

    def testBinaryClassesTrainWithTwoDimsWeight(self):
        self._testTrainWithTwoDimsWeight(n_classes=2)

    def testMultiClassesTrainWithTwoDimsWeight(self):
        self._testTrainWithTwoDimsWeight(n_classes=4)

    def _testTrainWithOneDimWeight(self, n_classes):
        batch_size = 20
        est = linear.LinearClassifier(feature_columns=(self._fc_lib.numeric_column('age'),), weight_column='w', n_classes=n_classes, model_dir=self._model_dir)
        data_rank_1 = np.array([0, 1])
        self.assertEqual((2,), data_rank_1.shape)
        train_input_fn = numpy_io.numpy_input_fn(x={'age': data_rank_1, 'w': data_rank_1}, y=data_rank_1, batch_size=batch_size, num_epochs=None, shuffle=True)
        est.train(train_input_fn, steps=200)
        self._assert_checkpoint(n_classes, 200)

    def testBinaryClassesTrainWithOneDimWeight(self):
        self._testTrainWithOneDimWeight(n_classes=2)

    def testMultiClassesTrainWithOneDimWeight(self):
        self._testTrainWithOneDimWeight(n_classes=4)

    def _testFromScratch(self, n_classes):
        label = 1
        age = 17
        mock_optimizer = self._mock_optimizer(expected_loss=-1 * math.log(1.0 / n_classes))
        est = linear.LinearClassifier(feature_columns=(self._fc_lib.numeric_column('age'),), n_classes=n_classes, optimizer=mock_optimizer, model_dir=self._model_dir)
        self.assertEqual(0, mock_optimizer.minimize.call_count)
        num_steps = 10
        est.train(input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
        self.assertEqual(1, mock_optimizer.minimize.call_count)
        self._assert_checkpoint(n_classes, expected_global_step=num_steps, expected_age_weight=[[0.0]] if n_classes == 2 else [[0.0] * n_classes], expected_bias=[0.0] if n_classes == 2 else [0.0] * n_classes)

    def testBinaryClassesFromScratch(self):
        self._testFromScratch(n_classes=2)

    def testMultiClassesFromScratch(self):
        self._testFromScratch(n_classes=4)

    def _testFromCheckpoint(self, n_classes):
        label = 1
        age = 17
        age_weight = [[2.0]] if n_classes == 2 else np.reshape(2.0 * np.array(list(range(n_classes)), dtype=np.float32), (1, n_classes))
        bias = [-35.0] if n_classes == 2 else [-35.0] * n_classes
        initial_global_step = 100
        with tf.Graph().as_default():
            tf.Variable(age_weight, name=AGE_WEIGHT_NAME)
            tf.Variable(bias, name=BIAS_NAME)
            tf.Variable(initial_global_step, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        if n_classes == 2:
            expected_loss = 1.3133
        else:
            logits = age_weight * age + bias
            logits_exp = np.exp(logits)
            softmax = logits_exp / logits_exp.sum()
            expected_loss = -1 * math.log(softmax[0, label])
        mock_optimizer = self._mock_optimizer(expected_loss=expected_loss)
        est = linear.LinearClassifier(feature_columns=(self._fc_lib.numeric_column('age'),), n_classes=n_classes, optimizer=mock_optimizer, model_dir=self._model_dir)
        self.assertEqual(0, mock_optimizer.minimize.call_count)
        num_steps = 10
        est.train(input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
        self.assertEqual(1, mock_optimizer.minimize.call_count)
        self._assert_checkpoint(n_classes, expected_global_step=initial_global_step + num_steps, expected_age_weight=age_weight, expected_bias=bias)

    def testBinaryClassesFromCheckpoint(self):
        self._testFromCheckpoint(n_classes=2)

    def testMultiClassesFromCheckpoint(self):
        self._testFromCheckpoint(n_classes=4)

    def _testFromCheckpointFloatLabels(self, n_classes):
        """Tests float labels for binary classification."""
        if n_classes > 2:
            return
        label = 0.8
        age = 17
        age_weight = [[2.0]]
        bias = [-35.0]
        initial_global_step = 100
        with tf.Graph().as_default():
            tf.Variable(age_weight, name=AGE_WEIGHT_NAME)
            tf.Variable(bias, name=BIAS_NAME)
            tf.Variable(initial_global_step, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        mock_optimizer = self._mock_optimizer(expected_loss=1.1132617)
        est = linear.LinearClassifier(feature_columns=(self._fc_lib.numeric_column('age'),), n_classes=n_classes, optimizer=mock_optimizer, model_dir=self._model_dir)
        self.assertEqual(0, mock_optimizer.minimize.call_count)
        num_steps = 10
        est.train(input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
        self.assertEqual(1, mock_optimizer.minimize.call_count)

    def testBinaryClassesFromCheckpointFloatLabels(self):
        self._testFromCheckpointFloatLabels(n_classes=2)

    def testMultiClassesFromCheckpointFloatLabels(self):
        self._testFromCheckpointFloatLabels(n_classes=4)

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

    def testBinaryClassesFromCheckpointMultiBatch(self):
        self._testFromCheckpointMultiBatch(n_classes=2)

    def testMultiClassesFromCheckpointMultiBatch(self):
        self._testFromCheckpointMultiBatch(n_classes=4)