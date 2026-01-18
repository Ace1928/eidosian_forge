from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def D_weighting(frequencies: _ScalarOrSequence[_FloatLike_co], *, min_db: Optional[float]=-80.0) -> Union[np.floating[Any], np.ndarray]:
    """Compute the D-weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    D_weighting : scalar or np.ndarray [shape=(n,)]
        ``D_weighting[i]`` is the D-weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    A_weighting
    B_weighting
    C_weighting

    Examples
    --------
    Get the D-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.D_weighting(freqs)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)', ylabel='Weighting (log10)',
    ...        title='D-Weighting of CQT frequencies')
    """
    f_sq = np.asanyarray(frequencies) ** 2.0
    const = np.array([0.0083046305, 1018.7, 1039.6, 3136.5, 3424, 282.7, 1160]) ** 2.0
    weights: np.ndarray = 20.0 * (0.5 * np.log10(f_sq) - np.log10(const[0]) + 0.5 * (+np.log10((const[1] - f_sq) ** 2 + const[2] * f_sq) - np.log10((const[3] - f_sq) ** 2 + const[4] * f_sq) - np.log10(const[5] + f_sq) - np.log10(const[6] + f_sq)))
    if min_db is None:
        return weights
    else:
        return np.maximum(min_db, weights)