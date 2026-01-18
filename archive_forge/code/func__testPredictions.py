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
def _testPredictions(self, n_classes, label_vocabulary, label_output_fn):
    """Tests predict when all variables are one-dimensional."""
    age = 1.0
    age_weight = [[-11.0]] if n_classes == 2 else np.reshape(-11.0 * np.array(list(range(n_classes)), dtype=np.float32), (1, n_classes))
    bias = [10.0] if n_classes == 2 else [10.0] * n_classes
    with tf.Graph().as_default():
        tf.Variable(age_weight, name=AGE_WEIGHT_NAME)
        tf.Variable(bias, name=BIAS_NAME)
        tf.Variable(100, name='global_step', dtype=tf.dtypes.int64)
        save_variables_to_ckpt(self._model_dir)
    est = self._linear_classifier_fn(feature_columns=(self._fc_lib.numeric_column('age'),), label_vocabulary=label_vocabulary, n_classes=n_classes, model_dir=self._model_dir)
    predict_input_fn = numpy_io.numpy_input_fn(x={'age': np.array([[age]])}, y=None, batch_size=1, num_epochs=1, shuffle=False)
    predictions = list(est.predict(input_fn=predict_input_fn))
    if n_classes == 2:
        scalar_logits = np.reshape(np.array(age_weight) * age + bias, (1,)).item()
        two_classes_logits = [0, scalar_logits]
        two_classes_logits_exp = np.exp(two_classes_logits)
        softmax = two_classes_logits_exp / two_classes_logits_exp.sum()
        expected_predictions = {'class_ids': [0], 'all_class_ids': [0, 1], 'classes': [label_output_fn(0)], 'all_classes': [label_output_fn(0), label_output_fn(1)], 'logistic': [sigmoid(np.array(scalar_logits))], 'logits': [scalar_logits], 'probabilities': softmax}
    else:
        onedim_logits = np.reshape(np.array(age_weight) * age + bias, (-1,))
        class_ids = onedim_logits.argmax()
        all_class_ids = list(range(len(onedim_logits)))
        logits_exp = np.exp(onedim_logits)
        softmax = logits_exp / logits_exp.sum()
        expected_predictions = {'class_ids': [class_ids], 'all_class_ids': all_class_ids, 'classes': [label_output_fn(class_ids)], 'all_classes': [label_output_fn(i) for i in all_class_ids], 'logits': onedim_logits, 'probabilities': softmax}
    self.assertEqual(1, len(predictions))
    self.assertEqual(expected_predictions['classes'], predictions[0]['classes'])
    expected_predictions.pop('classes')
    predictions[0].pop('classes')
    self.assertAllEqual(expected_predictions['all_classes'], predictions[0]['all_classes'])
    expected_predictions.pop('all_classes')
    predictions[0].pop('all_classes')
    self.assertAllClose(sorted_key_dict(expected_predictions), sorted_key_dict(predictions[0]))