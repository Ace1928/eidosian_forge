import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
def _apply_affine():
    for s in streamlines_gen:
        yield apply_affine(self._affine_to_apply, s)