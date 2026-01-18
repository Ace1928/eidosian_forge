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
class BaseDNNClassifierPredictTest(object):

    def __init__(self, dnn_classifier_fn, fc_impl=feature_column):
        self._dnn_classifier_fn = dnn_classifier_fn
        self._fc_impl = fc_impl

    def setUp(self):
        self._model_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self._model_dir:
            tf.compat.v1.summary.FileWriterCache.clear()
            shutil.rmtree(self._model_dir)

    def _test_one_dim(self, label_vocabulary, label_output_fn):
        """Asserts predictions for one-dimensional input and logits."""
        create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0], [1.0]], [0.3])), global_step=0, model_dir=self._model_dir)
        dnn_classifier = self._dnn_classifier_fn(hidden_units=(2, 2), label_vocabulary=label_vocabulary, feature_columns=(self._fc_impl.numeric_column('x'),), model_dir=self._model_dir)
        input_fn = numpy_io.numpy_input_fn(x={'x': np.array([[10.0]])}, batch_size=1, shuffle=False)
        predictions = next(dnn_classifier.predict(input_fn=input_fn))
        self.assertAllClose([-2.08], predictions[prediction_keys.PredictionKeys.LOGITS])
        self.assertAllClose([0.11105597], predictions[prediction_keys.PredictionKeys.LOGISTIC])
        self.assertAllClose([0.88894403, 0.11105597], predictions[prediction_keys.PredictionKeys.PROBABILITIES])
        self.assertAllClose([0], predictions[prediction_keys.PredictionKeys.CLASS_IDS])
        self.assertAllEqual([label_output_fn(0)], predictions[prediction_keys.PredictionKeys.CLASSES])

    def test_one_dim_without_label_vocabulary(self):
        self._test_one_dim(label_vocabulary=None, label_output_fn=lambda x: ('%s' % x).encode())

    def test_one_dim_with_label_vocabulary(self):
        n_classes = 2
        self._test_one_dim(label_vocabulary=['class_vocab_{}'.format(i) for i in range(n_classes)], label_output_fn=lambda x: ('class_vocab_%s' % x).encode())

    def _test_multi_dim_with_3_classes(self, label_vocabulary, label_output_fn):
        """Asserts predictions for multi-dimensional input and logits."""
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), global_step=0, model_dir=self._model_dir)
        dnn_classifier = self._dnn_classifier_fn(hidden_units=(2, 2), feature_columns=(self._fc_impl.numeric_column('x', shape=(2,)),), label_vocabulary=label_vocabulary, n_classes=3, model_dir=self._model_dir)
        input_fn = numpy_io.numpy_input_fn(x={'x': np.array([[10.0, 8.0]])}, batch_size=1, shuffle=False)
        predictions = next(dnn_classifier.predict(input_fn=input_fn))
        self.assertItemsEqual([prediction_keys.PredictionKeys.LOGITS, prediction_keys.PredictionKeys.PROBABILITIES, prediction_keys.PredictionKeys.CLASS_IDS, prediction_keys.PredictionKeys.CLASSES, prediction_keys.PredictionKeys.ALL_CLASS_IDS, prediction_keys.PredictionKeys.ALL_CLASSES], six.iterkeys(predictions))
        self.assertAllClose([-0.48, 0.48, 0.39], predictions[prediction_keys.PredictionKeys.LOGITS])
        self.assertAllClose([0.16670536, 0.4353838, 0.39791084], predictions[prediction_keys.PredictionKeys.PROBABILITIES])
        self.assertAllEqual([1], predictions[prediction_keys.PredictionKeys.CLASS_IDS])
        self.assertAllEqual([label_output_fn(1)], predictions[prediction_keys.PredictionKeys.CLASSES])

    def test_multi_dim_with_3_classes_but_no_label_vocab(self):
        self._test_multi_dim_with_3_classes(label_vocabulary=None, label_output_fn=lambda x: ('%s' % x).encode())

    def test_multi_dim_with_3_classes_and_label_vocab(self):
        n_classes = 3
        self._test_multi_dim_with_3_classes(label_vocabulary=['class_vocab_{}'.format(i) for i in range(n_classes)], label_output_fn=lambda x: ('class_vocab_%s' % x).encode())