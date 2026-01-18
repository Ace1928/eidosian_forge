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
def __bo_fold(d):
    """Compute the balanced, octave-folded interval.

    This maps intervals to the range [sqrt(2)/2, sqrt(2)).

    This is part of the FJS notation converter.
    It is equivalent to the `reb` function described in the FJS
    documentation, but with a simpler implementation.
    """
    return d * 2.0 ** (-np.round(np.log2(d)))