from abc import abstractmethod
from typing import (
from .config import registry
from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
def _make_mask(guesses, missing) -> Floats2d:
    xp = get_array_module(guesses)
    mask = xp.ones(guesses.shape, dtype='f')
    mask[missing] = 0
    return mask