import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatdim.register(Box)
@flatdim.register(MultiBinary)
def _flatdim_box_multibinary(space: Union[Box, MultiBinary]) -> int:
    return reduce(op.mul, space.shape, 1)