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
def _check_feature_shapes_compatible_with(features, compatible_with_name, compatible_with_value, ignore=None):
    """Checks all features are compatible with the given time-like feature."""
    if ignore is None:
        ignore = set()
    for name, value in features.items():
        if name in ignore:
            continue
        feature_shape = value.get_shape()
        if feature_shape.ndims is None:
            continue
        if feature_shape.ndims < 2:
            raise ValueError("Features must have shape (batch dimension, window size, ...) (got rank {} for feature '{}')".format(feature_shape.ndims, name))
        if not feature_shape[:2].is_compatible_with(compatible_with_value.get_shape()):
            raise ValueError("Features must have shape (batch dimension, window size, ...) where batch dimension and window size match the '{times_feature}' feature (got shape {feature_shape} for feature '{feature_name}' but shape {times_shape} for feature '{times_feature}')".format(times_feature=compatible_with_name, feature_shape=feature_shape, feature_name=name, times_shape=compatible_with_value.get_shape()))