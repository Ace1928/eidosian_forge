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