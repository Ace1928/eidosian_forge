from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
class FilteringOnlyStateManager(_OverridableStateManager):
    """State manager for models which use state only for filtering.

  Window-based models (ARModel) do not require state to be fed during training
  (instead requiring a specific window size). Rather than requiring a minimum
  window size for filtering, these models maintain this window in their state,
  and so need state to be fed.
  """

    def _define_loss_with_saved_state(self, model, features, mode):
        return model.define_loss(features, mode)