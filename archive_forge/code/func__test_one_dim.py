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