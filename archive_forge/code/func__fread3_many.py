import getpass
import time
import warnings
from collections import OrderedDict
import numpy as np
from ..openers import Opener
def _fread3_many(fobj, n):
    """Read 3-byte ints from an open binary file object.

    Parameters
    ----------
    fobj : file
        File descriptor

    Returns
    -------
    out : 1D array
        An array of 3 byte int
    """
    b1, b2, b3 = np.fromfile(fobj, '>u1', 3 * n).reshape(-1, 3).astype(int).T
    return (b1 << 16) + (b2 << 8) + b3