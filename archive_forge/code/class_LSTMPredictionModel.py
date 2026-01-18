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
class LSTMPredictionModel(tf.keras.models.Model):
    """A simple encoder/decoder model using an LSTM.

  This model does not operate on its own, but rather is a plugin to
  `ARModel`. See `ARModel`'s constructor documentation
  (`prediction_model_factory`) for a usage example.
  """

    def __init__(self, num_features, input_window_size, output_window_size, num_units=128):
        """Construct the LSTM prediction model.

    Args:
      num_features: number of input features per time step.
      input_window_size: Number of past time steps of data to look at when doing
        the regression.
      output_window_size: Number of future time steps to predict. Note that
        setting it to > 1 empirically seems to give a better fit.
      num_units: The number of units in the encoder and decoder LSTM cells.
    """
        super(LSTMPredictionModel, self).__init__()
        self._encoder = tf.keras.layers.LSTM(num_units, name='encoder', dtype=self.dtype, return_state=True)
        self._decoder = tf.keras.layers.LSTM(num_units, name='decoder', dtype=self.dtype, return_sequences=True)
        self._mean_transform = tf.keras.layers.Dense(num_features, name='mean_transform')
        self._covariance_transform = tf.keras.layers.Dense(num_features, name='covariance_transform')

    def call(self, input_window_features, output_window_features):
        """Compute predictions from input and output windows."""
        _, state_h, state_c = self._encoder(input_window_features)
        encoder_states = [state_h, state_c]
        decoder_output = self._decoder(output_window_features, initial_state=encoder_states)
        predicted_mean = self._mean_transform(decoder_output)
        predicted_covariance = gen_math_ops.exp(self._covariance_transform(decoder_output))
        return {'mean': predicted_mean, 'covariance': predicted_covariance}