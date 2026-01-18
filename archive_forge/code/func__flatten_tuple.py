import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatten.register(Tuple)
def _flatten_tuple(space, x) -> Union[tuple, np.ndarray]:
    if space.is_np_flattenable:
        return np.concatenate([flatten(s, x_part) for x_part, s in zip(x, space.spaces)])
    return tuple((flatten(s, x_part) for x_part, s in zip(x, space.spaces)))