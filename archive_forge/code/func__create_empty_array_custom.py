from collections import OrderedDict
from functools import singledispatch
from typing import Iterable, Union
import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@create_empty_array.register(Space)
def _create_empty_array_custom(space, n=1, fn=np.zeros):
    return None