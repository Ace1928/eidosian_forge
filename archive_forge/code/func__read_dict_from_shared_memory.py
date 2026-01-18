import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@read_from_shared_memory.register(Dict)
def _read_dict_from_shared_memory(space, shared_memory, n: int=1):
    return OrderedDict([(key, read_from_shared_memory(subspace, shared_memory[key], n=n)) for key, subspace in space.spaces.items()])