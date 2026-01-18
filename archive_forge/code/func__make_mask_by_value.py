from abc import abstractmethod
from typing import (
from .config import registry
from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
def _make_mask_by_value(truths, guesses, missing_value) -> Floats2d:
    xp = get_array_module(guesses)
    mask = xp.ones(guesses.shape, dtype='f')
    if missing_value is not None:
        if truths.ndim == 1:
            mask[truths == missing_value] = 0.0
        else:
            labels = xp.argmax(truths, axis=-1)
            mask[labels == missing_value] = 0.0
    return mask