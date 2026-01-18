import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@unflatten.register(Discrete)
def _unflatten_discrete(space: Discrete, x: np.ndarray) -> int:
    return int(space.start + np.nonzero(x)[0][0])