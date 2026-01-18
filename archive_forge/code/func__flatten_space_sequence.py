import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatten_space.register(Sequence)
def _flatten_space_sequence(space: Sequence) -> Sequence:
    return Sequence(flatten_space(space.feature_space))