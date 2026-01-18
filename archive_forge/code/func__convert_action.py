from gymnasium import core, spaces
import numpy as np
from ray.rllib.utils.annotations import PublicAPI
def _convert_action(self, action):
    action = action.astype(np.float64)
    true_delta = self._true_action_space.high - self._true_action_space.low
    norm_delta = self._norm_action_space.high - self._norm_action_space.low
    action = (action - self._norm_action_space.low) / norm_delta
    action = action * true_delta + self._true_action_space.low
    action = action.astype(np.float32)
    return action