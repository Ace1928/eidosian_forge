from collections import OrderedDict
from functools import singledispatch
from typing import Iterable, Union
import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@create_empty_array.register(Tuple)
def _create_empty_array_tuple(space, n=1, fn=np.zeros):
    return tuple((create_empty_array(subspace, n=n, fn=fn) for subspace in space.spaces))