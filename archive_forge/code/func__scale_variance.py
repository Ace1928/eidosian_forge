from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def _scale_variance(self, variance):
    """Scale variances according to stats (input scale -> model scale)."""
    if self._input_statistics is not None:
        return variance / self._input_statistics.overall_feature_moments.variance
    else:
        return variance