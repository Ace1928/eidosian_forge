import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@create_shared_memory.register(Box)
@create_shared_memory.register(Discrete)
@create_shared_memory.register(MultiDiscrete)
@create_shared_memory.register(MultiBinary)
def _create_base_shared_memory(space, n: int=1, ctx=mp):
    dtype = space.dtype.char
    if dtype in '?':
        dtype = c_bool
    return ctx.Array(dtype, n * int(np.prod(space.shape)))