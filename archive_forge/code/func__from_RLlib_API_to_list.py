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
def _from_RLlib_API_to_list(self, actions):
    """
        Format actions from dict of players to list of lists
        """
    actions = [actions[player_id] for player_id in self.players_ids]
    return actions