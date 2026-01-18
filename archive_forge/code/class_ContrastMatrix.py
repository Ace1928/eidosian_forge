from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
class ContrastMatrix(object):
    """A simple container for a matrix used for coding categorical factors.

    Attributes:

    .. attribute:: matrix

       A 2d ndarray, where each column corresponds to one column of the
       resulting design matrix, and each row contains the entries for a single
       categorical variable level. Usually n-by-n for a full rank coding or
       n-by-(n-1) for a reduced rank coding, though other options are
       possible.

    .. attribute:: column_suffixes

       A list of strings to be appended to the factor name, to produce the
       final column names. E.g. for treatment coding the entries will look
       like ``"[T.level1]"``.
    """

    def __init__(self, matrix, column_suffixes):
        self.matrix = np.asarray(matrix)
        self.column_suffixes = column_suffixes
        if self.matrix.shape[1] != len(column_suffixes):
            raise PatsyError("matrix and column_suffixes don't conform")
    __repr__ = repr_pretty_delegate

    def _repr_pretty_(self, p, cycle):
        repr_pretty_impl(p, self, [self.matrix, self.column_suffixes])
    __getstate__ = no_pickling