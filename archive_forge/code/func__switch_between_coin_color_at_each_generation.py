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
def _switch_between_coin_color_at_each_generation(self):
    self.red_coin = 1 - self.red_coin