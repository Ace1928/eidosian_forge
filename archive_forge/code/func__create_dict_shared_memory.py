import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@create_shared_memory.register(Dict)
def _create_dict_shared_memory(space, n=1, ctx=mp):
    return OrderedDict([(key, create_shared_memory(subspace, n=n, ctx=ctx)) for key, subspace in space.spaces.items()])