from collections import OrderedDict
import logging
import numpy as np
import gymnasium as gym
from typing import Any, List
from ray.rllib.utils.annotations import override, PublicAPI, DeveloperAPI
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.images import resize
from ray.rllib.utils.spaces.space_utils import convert_element_to_space_type
@DeveloperAPI
class RepeatedValuesPreprocessor(Preprocessor):
    """Pads and batches the variable-length list value."""

    @override(Preprocessor)
    def _init_shape(self, obs_space: gym.Space, options: dict) -> List[int]:
        assert isinstance(self._obs_space, Repeated)
        child_space = obs_space.child_space
        self.child_preprocessor = get_preprocessor(child_space)(child_space, self._options)
        size = 1 + self.child_preprocessor.size * obs_space.max_len
        return (size,)

    @override(Preprocessor)
    def transform(self, observation: TensorType) -> np.ndarray:
        array = np.zeros(self.shape)
        if isinstance(observation, list):
            for elem in observation:
                self.child_preprocessor.check_shape(elem)
        else:
            pass
        self.write(observation, array, 0)
        return array

    @override(Preprocessor)
    def write(self, observation: TensorType, array: np.ndarray, offset: int) -> None:
        if not isinstance(observation, (list, np.ndarray)):
            raise ValueError('Input for {} must be list type, got {}'.format(self, observation))
        elif len(observation) > self._obs_space.max_len:
            raise ValueError('Input {} exceeds max len of space {}'.format(observation, self._obs_space.max_len))
        array[offset] = len(observation)
        for i, elem in enumerate(observation):
            offset_i = offset + 1 + i * self.child_preprocessor.size
            self.child_preprocessor.write(elem, array, offset_i)