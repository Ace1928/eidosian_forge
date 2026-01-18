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
def _evaluate_ops(self, features):
    """Add ops for evaluation (aka filtering) to the graph."""
    spec = super(OneShotPredictionHead, self)._evaluate_ops(features)
    del spec.eval_metric_ops[feature_keys.State.STATE_TUPLE]
    return spec