import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override
@jit(nopython=True)
def _flatten_index(pos, grid_size):
    y_pos, x_pos = pos
    idx = grid_size * y_pos
    idx += x_pos
    return idx