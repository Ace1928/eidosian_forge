from collections import OrderedDict
from functools import singledispatch
from typing import Iterable, Union
import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@concatenate.register(Space)
def _concatenate_custom(space, items, out):
    return tuple(items)