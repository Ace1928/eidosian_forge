import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@unflatten.register(Text)
def _unflatten_text(space: Text, x: np.ndarray) -> str:
    return ''.join([space.character_list[val] for val in x if val < len(space.character_set)])