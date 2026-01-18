import copy
from typing import Sequence
import gym
from gym import spaces
def _filter_observation(self, observation):
    observation = type(observation)([(name, value) for name, value in observation.items() if name in self._filter_keys])
    return observation