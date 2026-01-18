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
@PublicAPI
class Preprocessor:
    """Defines an abstract observation preprocessor function.

    Attributes:
        shape (List[int]): Shape of the preprocessed output.
    """

    @PublicAPI
    def __init__(self, obs_space: gym.Space, options: dict=None):
        _legacy_patch_shapes(obs_space)
        self._obs_space = obs_space
        if not options:
            from ray.rllib.models.catalog import MODEL_DEFAULTS
            self._options = MODEL_DEFAULTS.copy()
        else:
            self._options = options
        self.shape = self._init_shape(obs_space, self._options)
        self._size = int(np.product(self.shape))
        self._i = 0
        self._obs_for_type_matching = self._obs_space.sample()

    @PublicAPI
    def _init_shape(self, obs_space: gym.Space, options: dict) -> List[int]:
        """Returns the shape after preprocessing."""
        raise NotImplementedError

    @PublicAPI
    def transform(self, observation: TensorType) -> np.ndarray:
        """Returns the preprocessed observation."""
        raise NotImplementedError

    def write(self, observation: TensorType, array: np.ndarray, offset: int) -> None:
        """Alternative to transform for more efficient flattening."""
        array[offset:offset + self._size] = self.transform(observation)

    def check_shape(self, observation: Any) -> None:
        """Checks the shape of the given observation."""
        if self._i % OBS_VALIDATION_INTERVAL == 0:
            if type(observation) is list and isinstance(self._obs_space, gym.spaces.Box):
                observation = np.array(observation).astype(np.float32)
            if not self._obs_space.contains(observation):
                observation = convert_element_to_space_type(observation, self._obs_for_type_matching)
            try:
                if not self._obs_space.contains(observation):
                    raise ValueError('Observation ({} dtype={}) outside given space ({})!'.format(observation, observation.dtype if isinstance(self._obs_space, gym.spaces.Box) else None, self._obs_space))
            except AttributeError as e:
                raise ValueError('Observation for a Box/MultiBinary/MultiDiscrete space should be an np.array, not a Python list.', observation) from e
        self._i += 1

    @property
    @PublicAPI
    def size(self) -> int:
        return self._size

    @property
    @PublicAPI
    def observation_space(self) -> gym.Space:
        obs_space = gym.spaces.Box(-1.0, 1.0, self.shape, dtype=np.float32)
        classes = (DictFlatteningPreprocessor, OneHotPreprocessor, RepeatedValuesPreprocessor, TupleFlatteningPreprocessor, AtariRamPreprocessor, GenericPixelPreprocessor)
        if isinstance(self, classes):
            obs_space.original_space = self._obs_space
        return obs_space