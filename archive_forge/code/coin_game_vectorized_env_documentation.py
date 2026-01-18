import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override

    Vectorized Coin Game environment.
    