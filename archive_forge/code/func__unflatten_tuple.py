import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@unflatten.register(Tuple)
def _unflatten_tuple(space: Tuple, x: Union[np.ndarray, tuple]) -> tuple:
    if space.is_np_flattenable:
        assert isinstance(x, np.ndarray), f'{space} is numpy-flattenable. Thus, you should only unflatten numpy arrays for this space. Got a {type(x)}'
        dims = np.asarray([flatdim(s) for s in space.spaces], dtype=np.int_)
        list_flattened = np.split(x, np.cumsum(dims[:-1]))
        return tuple((unflatten(s, flattened) for flattened, s in zip(list_flattened, space.spaces)))
    assert isinstance(x, tuple), f'{space} is not numpy-flattenable. Thus, you should only unflatten tuples for this space. Got a {type(x)}'
    return tuple((unflatten(s, flattened) for flattened, s in zip(x, space.spaces)))