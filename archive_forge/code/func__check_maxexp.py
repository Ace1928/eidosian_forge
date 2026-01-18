from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def _check_maxexp(np_type, maxexp):
    """True if fp type `np_type` seems to have `maxexp` maximum exponent

    We're testing "maxexp" as returned by numpy. This value is set to one
    greater than the maximum power of 2 that `np_type` can represent.

    Assumes base 2 representation.  Very crude check

    Parameters
    ----------
    np_type : numpy type specifier
        Any specifier for a numpy dtype
    maxexp : int
        Maximum exponent to test against

    Returns
    -------
    tf : bool
        True if `maxexp` is the correct maximum exponent, False otherwise.
    """
    dt = np.dtype(np_type)
    np_type = dt.type
    two = np_type(2).reshape((1,))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.isfinite(two ** (maxexp - 1)) and (not np.isfinite(two ** maxexp))