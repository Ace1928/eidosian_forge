from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.export import export_lib
class OneShotPredictionHead(TimeSeriesRegressionHead):
    """A time series head which exports a single stateless serving signature.

  The serving default signature exported by this head expects `times`, `values`,
  and any exogenous features, but no state. `values` has shape `[batch_size,
  filter_length, num_features]` and `times` has shape `[batch_size,
  total_length]`, where `total_length > filter_length`. Any exogenous features
  must have their shapes prefixed by the shape of the `times` feature.

  When serving, first performs filtering on the series up to `filter_length`
  starting from the default start state for the model, then computes predictions
  on the remainder of the series, returning them.

  Model state is neither accepted nor returned, so filtering must be performed
  each time predictions are requested when using this head.
  """

    def _check_predict_features(self, features):
        """Raises errors if features are not suitable for one-shot prediction."""
        if feature_keys.PredictionFeatures.TIMES not in features:
            raise ValueError("Expected a '{}' feature for prediction.".format(feature_keys.PredictionFeatures.TIMES))
        if feature_keys.TrainEvalFeatures.VALUES not in features:
            raise ValueError("Expected a '{}' feature for prediction.".format(feature_keys.TrainEvalFeatures.VALUES))
        if feature_keys.PredictionFeatures.STATE_TUPLE not in features:
            raise ValueError("Expected a '{}' feature for prediction.".format(feature_keys.PredictionFeatures.STATE_TUPLE))
        times_feature = features[feature_keys.PredictionFeatures.TIMES]
        if not times_feature.get_shape().is_compatible_with([None, None]):
            raise ValueError("Expected shape (batch dimension, window size) for feature '{}' (got shape {})".format(feature_keys.PredictionFeatures.TIMES, times_feature.get_shape()))
        _check_feature_shapes_compatible_with(features=features, compatible_with_name=feature_keys.PredictionFeatures.TIMES, compatible_with_value=times_feature, ignore=set([feature_keys.PredictionFeatures.STATE_TUPLE, feature_keys.TrainEvalFeatures.VALUES]))

    def _evaluate_ops(self, features):
        """Add ops for evaluation (aka filtering) to the graph."""
        spec = super(OneShotPredictionHead, self)._evaluate_ops(features)
        del spec.eval_metric_ops[feature_keys.State.STATE_TUPLE]
        return spec

    def _serving_ops(self, features):
        """Add ops for serving to the graph."""
        with tf.compat.v1.variable_scope('model', use_resource=True):
            filtering_features = {}
            prediction_features = {}
            values_length = tf.compat.v1.shape(features[feature_keys.FilteringFeatures.VALUES])[1]
            for key, value in features.items():
                if key == feature_keys.State.STATE_TUPLE:
                    continue
                if key == feature_keys.FilteringFeatures.VALUES:
                    filtering_features[key] = value
                else:
                    filtering_features[key] = value[:, :values_length]
                    prediction_features[key] = value[:, values_length:]
            cold_filtering_outputs = self.model.define_loss(features=filtering_features, mode=estimator_lib.ModeKeys.EVAL)
            prediction_features[feature_keys.State.STATE_TUPLE] = cold_filtering_outputs.end_state
        with tf.compat.v1.variable_scope('model', reuse=True):
            prediction_outputs = self.model.predict(features=prediction_features)
        return estimator_lib.EstimatorSpec(mode=estimator_lib.ModeKeys.PREDICT, export_outputs={feature_keys.SavedModelLabels.PREDICT: _NoStatePredictOutput(prediction_outputs)}, predictions={})