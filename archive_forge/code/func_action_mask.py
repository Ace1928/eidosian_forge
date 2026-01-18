from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import numpy as np
from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
def action_mask(self, state: int):
    """Computes an action mask for the action space using the state information."""
    mask = np.zeros(6, dtype=np.int8)
    taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
    if taxi_row < 4:
        mask[0] = 1
    if taxi_row > 0:
        mask[1] = 1
    if taxi_col < 4 and self.desc[taxi_row + 1, 2 * taxi_col + 2] == b':':
        mask[2] = 1
    if taxi_col > 0 and self.desc[taxi_row + 1, 2 * taxi_col] == b':':
        mask[3] = 1
    if pass_loc < 4 and (taxi_row, taxi_col) == self.locs[pass_loc]:
        mask[4] = 1
    if pass_loc == 4 and ((taxi_row, taxi_col) == self.locs[dest_idx] or (taxi_row, taxi_col) in self.locs):
        mask[5] = 1
    return mask