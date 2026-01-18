from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
from absl import logging
import tensorflow as tf
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _extract_sample_weight_tensors(features):
    if isinstance(features, dict) and set(features.keys()) == {'features', 'sample_weights'}:
        feature_tensor = features['features']
        sample_weight_tensors = features['sample_weights']
    else:
        feature_tensor = features
        sample_weight_tensors = None
    return (feature_tensor, sample_weight_tensors)