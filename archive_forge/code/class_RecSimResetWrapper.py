from collections import OrderedDict
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, MultiDiscrete
from gymnasium.wrappers import EnvCompatibility
import numpy as np
from recsim.document import AbstractDocumentSampler
from recsim.simulator import environment, recsim_gym
from recsim.user import AbstractUserModel, AbstractResponse
from typing import Callable, List, Optional, Type
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.gym import convert_old_gym_space_to_gymnasium_space
from ray.rllib.utils.spaces.space_utils import convert_element_to_space_type
class RecSimResetWrapper(gym.Wrapper):
    """Fix RecSim environment's reset() and close() function

    RecSim's reset() function returns an observation without the "response"
    field, breaking RLlib's check. This wrapper fixes that by assigning a
    random "response".

    RecSim's close() function raises NotImplementedError. We change the
    behavior to doing nothing.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._sampled_obs = self.env.observation_space.sample()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset()
        obs['response'] = self.env.observation_space['response'].sample()
        obs = convert_element_to_space_type(obs, self._sampled_obs)
        return (obs, info)

    def close(self):
        pass