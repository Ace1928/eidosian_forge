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
def _legacy_patch_shapes(space: gym.Space) -> List[int]:
    """Assigns shapes to spaces that don't have shapes.

    This is only needed for older gym versions that don't set shapes properly
    for Tuple and Discrete spaces.
    """
    if not hasattr(space, 'shape'):
        if isinstance(space, gym.spaces.Discrete):
            space.shape = ()
        elif isinstance(space, gym.spaces.Tuple):
            shapes = []
            for s in space.spaces:
                shape = _legacy_patch_shapes(s)
                shapes.append(shape)
            space.shape = tuple(shapes)
    return space.shape