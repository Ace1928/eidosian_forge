from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import distributions
from tensorflow.python.ops import gen_math_ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned.timeseries import model
from tensorflow_estimator.python.estimator.canned.timeseries import model_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import PredictionFeatures
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def _output_window_predictions(self, input_window_features, output_window_features):
    with self._model_scope:
        predictions = self._model_instance(input_window_features, output_window_features)
        result_shape = [None, self.output_window_size, self.num_features]
        for v in predictions.values():
            v.set_shape(result_shape)
        return predictions