from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Iterator
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@batch_space.register(MultiDiscrete)
def _batch_space_multidiscrete(space, n=1):
    repeats = tuple([n] + [1] * space.nvec.ndim)
    high = np.tile(space.nvec, repeats) - 1
    return Box(low=np.zeros_like(high), high=high, dtype=space.dtype, seed=deepcopy(space.np_random))