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
def _move_players(self, actions):
    self.red_pos = (self.red_pos + self.MOVES[actions[0]]) % self.grid_size
    self.blue_pos = (self.blue_pos + self.MOVES[actions[1]]) % self.grid_size