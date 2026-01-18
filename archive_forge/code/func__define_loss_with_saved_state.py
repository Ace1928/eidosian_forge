from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
def _define_loss_with_saved_state(self, model, features, mode):
    return model.define_loss(features, mode)