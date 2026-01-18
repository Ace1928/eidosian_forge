import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@read_from_shared_memory.register(Box)
@read_from_shared_memory.register(Discrete)
@read_from_shared_memory.register(MultiDiscrete)
@read_from_shared_memory.register(MultiBinary)
def _read_base_from_shared_memory(space, shared_memory, n: int=1):
    return np.frombuffer(shared_memory.get_obj(), dtype=space.dtype).reshape((n,) + space.shape)