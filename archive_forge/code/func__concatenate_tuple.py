from collections import OrderedDict
from functools import singledispatch
from typing import Iterable, Union
import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@concatenate.register(Tuple)
def _concatenate_tuple(space, items, out):
    return tuple((concatenate(subspace, [item[i] for item in items], out[i]) for i, subspace in enumerate(space.spaces)))