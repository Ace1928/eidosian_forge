from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def Z_weighting(frequencies: Sized, *, min_db: Optional[float]=None) -> np.ndarray:
    """Apply no weighting curve (aka Z-weighting).

    This function behaves similarly to `A_weighting`, `B_weighting`, etc.,
    but all frequencies are equally weighted.
    An optional threshold `min_db` can still be used to clip energies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    Z_weighting : scalar or np.ndarray [shape=(n,)]
        ``Z_weighting[i]`` is the Z-weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    A_weighting
    B_weighting
    C_weighting
    D_weighting
    """
    weights = np.zeros(len(frequencies))
    if min_db is None:
        return weights
    else:
        return np.maximum(min_db, weights)