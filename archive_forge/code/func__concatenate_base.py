from collections import OrderedDict
from functools import singledispatch
from typing import Iterable, Union
import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@concatenate.register(Box)
@concatenate.register(Discrete)
@concatenate.register(MultiDiscrete)
@concatenate.register(MultiBinary)
def _concatenate_base(space, items, out):
    return np.stack(items, axis=0, out=out)