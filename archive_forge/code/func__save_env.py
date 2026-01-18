import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override
def _save_env(self):
    env_save_state = {'red_pos': self.red_pos, 'blue_pos': self.blue_pos, 'coin_pos': self.coin_pos, 'red_coin': self.red_coin, 'grid_size': self.grid_size, 'asymmetric': self.asymmetric, 'batch_size': self.batch_size, 'step_count_in_current_episode': self.step_count_in_current_episode, 'max_steps': self.max_steps, 'red_pick': self.red_pick, 'red_pick_own': self.red_pick_own, 'blue_pick': self.blue_pick, 'blue_pick_own': self.blue_pick_own, 'both_players_can_pick_the_same_coin': self.both_players_can_pick_the_same_coin}
    return copy.deepcopy(env_save_state)