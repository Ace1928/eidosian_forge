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
class TimeSeriesRegressionHead(head_lib._Head):
    """Determines input and output signatures for a time series model."""

    def __init__(self, model, state_manager, optimizer, input_statistics_generator=None, name=None):
        """Creates a `_Head` for time series regression.

    Args:
      model: A model for time series regression.
      state_manager: A state manager.
      optimizer: An optimizer.
      input_statistics_generator: A input statistics generator.
      name: An optional name for the model.
    """
        self.model = model
        self.state_manager = state_manager
        self.optimizer = optimizer
        self.input_statistics_generator = input_statistics_generator
        self._name = name

    @property
    def name(self):
        return self._name

    def create_loss(self, features, mode, logits=None, labels=None):
        """See `_Head`."""
        model_outputs = self.state_manager.define_loss(self.model, features, mode)
        tf.compat.v1.summary.scalar(head_lib._summary_key(self._name, metric_keys.MetricKeys.LOSS), model_outputs.loss)
        return model_outputs

    @property
    def logits_dimension(self):
        """See `_Head`."""
        return 1

    def _train_ops(self, features):
        """Add training ops to the graph."""
        mode = estimator_lib.ModeKeys.TRAIN
        with tf.compat.v1.variable_scope('model', use_resource=True):
            model_outputs = self.create_loss(features, mode)
        train_op = self.optimizer.minimize(model_outputs.loss, global_step=tf.compat.v1.train.get_global_step())
        return estimator_lib.EstimatorSpec(loss=model_outputs.loss, mode=mode, train_op=train_op)

    def _evaluate_ops(self, features):
        """Add ops for evaluation (aka filtering) to the graph."""
        mode = estimator_lib.ModeKeys.EVAL
        with tf.compat.v1.variable_scope('model', use_resource=True):
            model_outputs = self.create_loss(features, mode)
        metrics = {}
        for prediction_key, prediction_value in model_outputs.predictions.items():
            metrics[prediction_key] = _identity_metric_single(prediction_key, prediction_value)
        metrics[feature_keys.FilteringResults.TIMES] = _identity_metric_single(feature_keys.FilteringResults.TIMES, model_outputs.prediction_times)
        metrics[feature_keys.FilteringResults.STATE_TUPLE] = _identity_metric_nested(feature_keys.FilteringResults.STATE_TUPLE, model_outputs.end_state)
        metrics[metric_keys.MetricKeys.LOSS_MEAN] = tf.compat.v1.metrics.mean(model_outputs.loss, name='average_loss')
        return estimator_lib.EstimatorSpec(loss=model_outputs.loss, mode=mode, eval_metric_ops=metrics, predictions=model_outputs.predictions)

    def _predict_ops(self, features):
        """Add ops for prediction to the graph."""
        with tf.compat.v1.variable_scope('model', use_resource=True):
            prediction = self.model.predict(features=features)
        prediction[feature_keys.PredictionResults.TIMES] = features[feature_keys.PredictionFeatures.TIMES]
        return estimator_lib.EstimatorSpec(predictions=prediction, mode=estimator_lib.ModeKeys.PREDICT)

    def _serving_ops(self, features):
        """Add ops for serving to the graph."""
        with tf.compat.v1.variable_scope('model', use_resource=True):
            prediction_outputs = self.model.predict(features=features)
        with tf.compat.v1.variable_scope('model', reuse=True):
            filtering_outputs = self.create_loss(features, estimator_lib.ModeKeys.EVAL)
        with tf.compat.v1.variable_scope('model', reuse=True):
            no_state_features = {k: v for k, v in features.items() if not k.startswith(feature_keys.State.STATE_PREFIX)}
            cold_filtering_outputs = self.model.define_loss(features=no_state_features, mode=estimator_lib.ModeKeys.EVAL)
        return estimator_lib.EstimatorSpec(mode=estimator_lib.ModeKeys.PREDICT, export_outputs={feature_keys.SavedModelLabels.PREDICT: export_lib.PredictOutput(prediction_outputs), feature_keys.SavedModelLabels.FILTER: export_lib.PredictOutput(state_to_dictionary(filtering_outputs.end_state)), feature_keys.SavedModelLabels.COLD_START_FILTER: _NoStatePredictOutput(state_to_dictionary(cold_filtering_outputs.end_state))}, predictions={})

    def _convert_feature_to_tensor(self, name, value):
        """Casts features to the correct dtype based on their name."""
        if name in [feature_keys.TrainEvalFeatures.TIMES, feature_keys.PredictionFeatures.TIMES]:
            return tf.cast(value, tf.dtypes.int64)
        if name == feature_keys.TrainEvalFeatures.VALUES:
            return tf.cast(value, self.model.dtype)
        if name == feature_keys.PredictionFeatures.STATE_TUPLE:
            return value
        return tf.compat.v1.convert_to_tensor_or_sparse_tensor(value)

    def _gather_state(self, features):
        """Returns `features` with state packed, indicates if packing was done."""
        prefixed_state_re = re.compile('^' + feature_keys.State.STATE_PREFIX + '_(\\d+)$')
        numbered_state = []
        for key, tensor in features.items():
            search_result = prefixed_state_re.search(key)
            if search_result:
                numbered_state.append((int(search_result.group(1)), key, tensor))
        if not numbered_state:
            return (features, False)
        features = features.copy()
        for _, key, _ in numbered_state:
            del features[key]
        numbered_state.sort(key=lambda number, *_: number)
        features[feature_keys.State.STATE_TUPLE] = tf.nest.pack_sequence_as(structure=self.model.get_start_state(), flat_sequence=[tensor for _, _, tensor in numbered_state])
        return (features, True)

    def _check_predict_features(self, features):
        """Raises errors if features are not suitable for prediction."""
        if feature_keys.PredictionFeatures.TIMES not in features:
            raise ValueError("Expected a '{}' feature for prediction.".format(feature_keys.PredictionFeatures.TIMES))
        if feature_keys.PredictionFeatures.STATE_TUPLE not in features:
            raise ValueError("Expected a '{}' feature for prediction.".format(feature_keys.PredictionFeatures.STATE_TUPLE))
        times_feature = features[feature_keys.PredictionFeatures.TIMES]
        if not times_feature.get_shape().is_compatible_with([None, None]):
            raise ValueError("Expected shape (batch dimension, window size) for feature '{}' (got shape {})".format(feature_keys.PredictionFeatures.TIMES, times_feature.get_shape()))
        _check_feature_shapes_compatible_with(features=features, compatible_with_name=feature_keys.PredictionFeatures.TIMES, compatible_with_value=times_feature, ignore=set([feature_keys.PredictionFeatures.STATE_TUPLE]))

    def create_estimator_spec(self, features, mode, labels=None):
        """Performs basic error checking and returns an EstimatorSpec."""
        with ops.name_scope(self._name, 'head'):
            if labels is not None and (not (isinstance(labels, dict) and labels == {})):
                raise ValueError("The model received a `labels`, which is not supported. Pass '{}' and '{}' as features.".format(feature_keys.TrainEvalFeatures.TIMES, feature_keys.TrainEvalFeatures.VALUES))
            del labels
            features = {name: self._convert_feature_to_tensor(name=name, value=value) for name, value in features.items()}
            if self.input_statistics_generator is not None:
                input_statistics = self.input_statistics_generator.initialize_graph(features, update_statistics=mode == estimator_lib.ModeKeys.TRAIN)
            else:
                input_statistics = None
            self.model.initialize_graph(input_statistics=input_statistics)
            features, passed_flat_state = self._gather_state(features)
            if mode == estimator_lib.ModeKeys.TRAIN or mode == estimator_lib.ModeKeys.EVAL:
                _check_train_eval_features(features, self.model)
            elif mode == estimator_lib.ModeKeys.PREDICT:
                self._check_predict_features(features)
            else:
                raise ValueError("Unknown mode '{}' passed to model_fn.".format(mode))
            self.state_manager.initialize_graph(model=self.model, input_statistics=input_statistics)
            if mode == estimator_lib.ModeKeys.TRAIN:
                return self._train_ops(features)
            elif mode == estimator_lib.ModeKeys.EVAL:
                return self._evaluate_ops(features)
            elif mode == estimator_lib.ModeKeys.PREDICT and (not passed_flat_state):
                return self._predict_ops(features)
            elif mode == estimator_lib.ModeKeys.PREDICT and passed_flat_state:
                return self._serving_ops(features)