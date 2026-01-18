import numpy as np
import gymnasium as gym
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import TensorType, List, Union, ModelConfigDict
Returns the required shape of an input parameter tensor for a
        particular action space and an optional dict of distribution-specific
        options.

        Args:
            action_space (gym.Space): The action space this distribution will
                be used for, whose shape attributes will be used to determine
                the required shape of the input parameter tensor.
            model_config: Model's config dict (as defined in catalog.py)

        Returns:
            model_output_shape (int or np.ndarray of ints): size of the
                required input vector (minus leading batch dimension).
        