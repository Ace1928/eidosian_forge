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
def _get_obs_invariant_to_the_player_trained(self, observation):
    """
        We want to be able to use a policy trained as player 1,
        for evaluation as player 2 and vice versa.
        """
    player_red_observation = observation
    player_blue_observation = copy.deepcopy(observation)
    player_blue_observation[..., 0] = observation[..., 1]
    player_blue_observation[..., 1] = observation[..., 0]
    player_blue_observation[..., 2] = observation[..., 3]
    player_blue_observation[..., 3] = observation[..., 2]
    return [player_red_observation, player_blue_observation]