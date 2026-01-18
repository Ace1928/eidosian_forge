import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatten.register(Box)
@flatten.register(MultiBinary)
def _flatten_box_multibinary(space, x) -> np.ndarray:
    return np.asarray(x, dtype=space.dtype).flatten()