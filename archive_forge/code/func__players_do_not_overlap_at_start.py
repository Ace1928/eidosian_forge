import copy
import gymnasium as gym
import logging
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import override
from typing import Dict, Optional
from ray.rllib.examples.env.utils.interfaces import InfoAccumulationInterface
def _players_do_not_overlap_at_start(self):
    while self._same_pos(self.red_pos, self.blue_pos):
        self.blue_pos = self.np_random.integers(self.grid_size, size=2)