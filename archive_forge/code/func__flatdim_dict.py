import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatdim.register(Dict)
def _flatdim_dict(space: Dict) -> int:
    if space.is_np_flattenable:
        return sum((flatdim(s) for s in space.spaces.values()))
    raise ValueError(f'{space} cannot be flattened to a numpy array, probably because it contains a `Graph` or `Sequence` subspace')