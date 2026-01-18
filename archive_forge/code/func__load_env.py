import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override
def _load_env(self, env_state):
    for k, v in env_state.items():
        self.__setattr__(k, v)