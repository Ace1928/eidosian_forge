from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Iterator
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@iterate.register(Box)
@iterate.register(MultiDiscrete)
@iterate.register(MultiBinary)
def _iterate_base(space, items):
    try:
        return iter(items)
    except TypeError:
        raise TypeError(f'Unable to iterate over the following elements: {items}')