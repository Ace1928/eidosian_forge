from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Iterator
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@batch_space.register(Space)
def _batch_space_custom(space, n=1):
    batched_space = Tuple(tuple((deepcopy(space) for _ in range(n))), seed=deepcopy(space.np_random))
    new_seeds = list(map(int, batched_space.np_random.integers(0, 100000000.0, n)))
    batched_space.seed(new_seeds)
    return batched_space