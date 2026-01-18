from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
class PassthroughStateManager(object):
    """A minimal wrapper for models which do not need state management."""

    def __init__(self):
        self._input_statistics = None
        self._graph_initialized = False

    def initialize_graph(self, model, input_statistics=None):
        """Adds required operations to the graph."""
        del model
        self._graph_initialized = True
        self._input_statistics = input_statistics

    def define_loss(self, model, features, mode):
        """Wrap "model" with StateManager-specific operations.

    Args:
      model: The model (inheriting from TimeSeriesModel) to manage state for.
      features: A dictionary with the following key/value pairs:
        feature_keys.TrainEvalFeatures.TIMES: A [batch size x window size]
          Tensor with times for each observation.
        feature_keys.TrainEvalFeatures.VALUES: A [batch size x window size x num
          features] Tensor with values for each observation.
      mode: The tf.estimator.ModeKeys mode to use (TRAIN or EVAL).

    Returns:
      A ModelOutputs object.
    Raises:
      ValueError: If start state was specified.
    """
        if feature_keys.State.STATE_TUPLE in features:
            raise ValueError('Overriding start state is not supported for this model.')
        return model.define_loss(features, mode)