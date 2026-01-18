import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@write_to_shared_memory.register(Box)
@write_to_shared_memory.register(Discrete)
@write_to_shared_memory.register(MultiDiscrete)
@write_to_shared_memory.register(MultiBinary)
def _write_base_to_shared_memory(space, index, value, shared_memory):
    size = int(np.prod(space.shape))
    destination = np.frombuffer(shared_memory.get_obj(), dtype=space.dtype)
    np.copyto(destination[index * size:(index + 1) * size], np.asarray(value, dtype=space.dtype).flatten())