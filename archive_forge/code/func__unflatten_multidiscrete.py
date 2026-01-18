import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@unflatten.register(MultiDiscrete)
def _unflatten_multidiscrete(space: MultiDiscrete, x: np.ndarray) -> np.ndarray:
    offsets = np.zeros((space.nvec.size + 1,), dtype=space.dtype)
    offsets[1:] = np.cumsum(space.nvec.flatten())
    indices, = cast(type(offsets[:-1]), np.nonzero(x))
    return np.asarray(indices - offsets[:-1], dtype=space.dtype).reshape(space.shape)