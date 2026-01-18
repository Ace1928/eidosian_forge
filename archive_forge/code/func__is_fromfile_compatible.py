import os
import numpy as np
from numpy import (asarray, real, imag, conj, zeros, ndarray, concatenate,
from scipy.sparse import coo_matrix, issparse
def _is_fromfile_compatible(stream):
    """
    Check whether `stream` is compatible with numpy.fromfile.

    Passing a gzipped file object to ``fromfile/fromstring`` doesn't work with
    Python 3.
    """
    bad_cls = []
    try:
        import gzip
        bad_cls.append(gzip.GzipFile)
    except ImportError:
        pass
    try:
        import bz2
        bad_cls.append(bz2.BZ2File)
    except ImportError:
        pass
    bad_cls = tuple(bad_cls)
    return not isinstance(stream, bad_cls)