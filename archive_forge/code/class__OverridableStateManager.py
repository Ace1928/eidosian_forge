from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
class _OverridableStateManager(PassthroughStateManager):
    """Base class for state managers which support overriding model state."""

    @abc.abstractmethod
    def _define_loss_with_saved_state(self, model, features, mode):
        pass

    def define_loss(self, model, features, mode):
        """Switches between explicit start state and managed state."""
        if feature_keys.FilteringFeatures.STATE_TUPLE in features:
            if mode == estimator_lib.ModeKeys.TRAIN:
                raise ValueError('Overriding saved state for training is not supported (but a value for feature {} was specified).'.format(feature_keys.FilteringFeatures.STATE_TUPLE))
            start_state = features[feature_keys.FilteringFeatures.STATE_TUPLE]
            del features[feature_keys.FilteringFeatures.STATE_TUPLE]
            return model.get_batch_loss(features=features, mode=mode, state=start_state)
        else:
            return self._define_loss_with_saved_state(model=model, features=features, mode=mode)