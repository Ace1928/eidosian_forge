from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def _check_nmant(np_type, nmant):
    """True if fp type `np_type` seems to have `nmant` significand digits

    Note 'digits' does not include implicit digits.  And in fact if there are
    no implicit digits, the `nmant` number is one less than the actual digits.
    Assumes base 2 representation.

    Parameters
    ----------
    np_type : numpy type specifier
        Any specifier for a numpy dtype
    nmant : int
        Number of digits to test against

    Returns
    -------
    tf : bool
        True if `nmant` is the correct number of significand digits, false
        otherwise
    """
    np_type = np.dtype(np_type).type
    max_contig = np_type(2 ** (nmant + 1))
    tests = max_contig + np.array([-2, -1, 0, 1, 2], dtype=np_type)
    return np.all(tests - max_contig == [-2, -1, 0, 0, 2])