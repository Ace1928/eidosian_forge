import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@unflatten.register(Sequence)
def _unflatten_sequence(space: Sequence, x: tuple) -> tuple:
    return tuple((unflatten(space.feature_space, item) for item in x))