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
def _load_config(self, config):
    self.players_ids = config.get('players_ids', ['player_red', 'player_blue'])
    self.max_steps = config.get('max_steps', 20)
    self.grid_size = config.get('grid_size', 3)
    self.output_additional_info = config.get('output_additional_info', True)
    self.asymmetric = config.get('asymmetric', False)
    self.both_players_can_pick_the_same_coin = config.get('both_players_can_pick_the_same_coin', True)