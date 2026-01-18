import re
import numpy as np
from numba import jit
from .intervals import INTERVALS
from .._cache import cache
from ..util.exceptions import ParameterError
from typing import Dict, List, Union, overload
from ..util.decorators import vectorize
from .._typing import _ScalarOrSequence, _FloatLike_co, _SequenceLike
@jit(nopython=True, nogil=True, cache=True)
def __o_fold(d):
    """Compute the octave-folded interval.

    This maps intervals to the range [1, 2).

    This is part of the FJS notation converter.
    It is equivalent to the `red` function described in the FJS
    documentation.
    """
    return d * 2.0 ** (-np.floor(np.log2(d)))