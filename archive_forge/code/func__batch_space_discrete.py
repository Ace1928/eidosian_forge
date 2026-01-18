from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Iterator
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@batch_space.register(Discrete)
def _batch_space_discrete(space, n=1):
    if space.start == 0:
        return MultiDiscrete(np.full((n,), space.n, dtype=space.dtype), dtype=space.dtype, seed=deepcopy(space.np_random))
    else:
        return Box(low=space.start, high=space.start + space.n - 1, shape=(n,), dtype=space.dtype, seed=deepcopy(space.np_random))