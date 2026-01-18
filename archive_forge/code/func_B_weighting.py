from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def B_weighting(frequencies: _ScalarOrSequence[_FloatLike_co], *, min_db: Optional[float]=-80.0) -> Union[np.floating[Any], np.ndarray]:
    """Compute the B-weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    B_weighting : scalar or np.ndarray [shape=(n,)]
        ``B_weighting[i]`` is the B-weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    A_weighting
    C_weighting
    D_weighting

    Examples
    --------
    Get the B-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.B_weighting(freqs)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)',
    ...        ylabel='Weighting (log10)',
    ...        title='B-Weighting of CQT frequencies')
    """
    f_sq = np.asanyarray(frequencies) ** 2.0
    const = np.array([12194.217, 20.598997, 158.48932]) ** 2.0
    weights: np.ndarray = 0.17 + 20.0 * (np.log10(const[0]) + 1.5 * np.log10(f_sq) - np.log10(f_sq + const[0]) - np.log10(f_sq + const[1]) - 0.5 * np.log10(f_sq + const[2]))
    return weights if min_db is None else np.maximum(min_db, weights)