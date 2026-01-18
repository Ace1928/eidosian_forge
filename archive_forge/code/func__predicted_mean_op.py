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
def _predicted_mean_op(self, activations):
    activation, activation_size = activations[-1]
    predicted_mean = model_utils.fully_connected(activation, activation_size, self.output_window_size * self.num_features, name='predicted_mean', activation=None)
    return tf.reshape(predicted_mean, [-1, self.output_window_size, self.num_features])