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
class TupleFlatteningPreprocessor(Preprocessor):
    """Preprocesses each tuple element, then flattens it all into a vector.

    RLlib models will unpack the flattened output before _build_layers_v2().
    """

    @override(Preprocessor)
    def _init_shape(self, obs_space: gym.Space, options: dict) -> List[int]:
        assert isinstance(self._obs_space, gym.spaces.Tuple)
        size = 0
        self.preprocessors = []
        for i in range(len(self._obs_space.spaces)):
            space = self._obs_space.spaces[i]
            logger.debug('Creating sub-preprocessor for {}'.format(space))
            preprocessor_class = get_preprocessor(space)
            if preprocessor_class is not None:
                preprocessor = preprocessor_class(space, self._options)
                size += preprocessor.size
            else:
                preprocessor = None
                size += int(np.product(space.shape))
            self.preprocessors.append(preprocessor)
        return (size,)

    @override(Preprocessor)
    def transform(self, observation: TensorType) -> np.ndarray:
        self.check_shape(observation)
        array = np.zeros(self.shape, dtype=np.float32)
        self.write(observation, array, 0)
        return array

    @override(Preprocessor)
    def write(self, observation: TensorType, array: np.ndarray, offset: int) -> None:
        assert len(observation) == len(self.preprocessors), observation
        for o, p in zip(observation, self.preprocessors):
            p.write(o, array, offset)
            offset += p.size