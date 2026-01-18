from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def _check_graph_initialized(self):
    if not self._graph_initialized:
        raise ValueError('TimeSeriesModels require initialize_graph() to be called before use. This defines variables and ops in the default graph, and allows Tensor-valued input statistics to be specified.')