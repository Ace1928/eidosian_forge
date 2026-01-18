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
class AsymCoinGame(CoinGame):
    NAME = 'AsymCoinGame'

    def __init__(self, config: Optional[dict]=None):
        if config is None:
            config = {}
        if 'asymmetric' in config:
            assert config['asymmetric']
        else:
            config['asymmetric'] = True
        super().__init__(config)