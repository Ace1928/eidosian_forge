import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
def _set_streamlines(self, value):
    if value is not None and (not callable(value)):
        msg = '`streamlines` must be a generator function. That is a function which, when called, returns an instantiated generator.'
        raise TypeError(msg)
    self._streamlines = value