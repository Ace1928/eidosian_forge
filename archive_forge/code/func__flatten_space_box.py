import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatten_space.register(Box)
def _flatten_space_box(space: Box) -> Box:
    return Box(space.low.flatten(), space.high.flatten(), dtype=space.dtype)