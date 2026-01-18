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
def _randomize_color_and_player_positions(self):
    self.red_coin = self.np_random.integers(low=0, high=2)
    self.red_pos = self.np_random.integers(low=0, high=self.grid_size, size=(2,))
    self.blue_pos = self.np_random.integers(low=0, high=self.grid_size, size=(2,))
    self.coin_pos = np.zeros(shape=(2,), dtype=np.int8)
    self._players_do_not_overlap_at_start()