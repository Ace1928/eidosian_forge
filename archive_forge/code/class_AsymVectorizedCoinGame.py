import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override
class AsymVectorizedCoinGame(VectorizedCoinGame):
    NAME = 'AsymCoinGame'

    def __init__(self, config=None):
        if config is None:
            config = {}
        if 'asymmetric' in config:
            assert config['asymmetric']
        else:
            config['asymmetric'] = True
        super().__init__(config)