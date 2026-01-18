import logging
from abc import ABC
from collections import Iterable
from typing import Dict, Optional
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.examples.env.utils.interfaces import InfoAccumulationInterface
from ray.rllib.examples.env.utils.mixins import (
def _get_players_rewards(self, action_player_0: int, action_player_1: int):
    return [self.PAYOUT_MATRIX[action_player_0][action_player_1][0], self.PAYOUT_MATRIX[action_player_0][action_player_1][1]]