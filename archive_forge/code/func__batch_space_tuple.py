from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Iterator
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@batch_space.register(Tuple)
def _batch_space_tuple(space, n=1):
    return Tuple(tuple((batch_space(subspace, n=n) for subspace in space.spaces)), seed=deepcopy(space.np_random))