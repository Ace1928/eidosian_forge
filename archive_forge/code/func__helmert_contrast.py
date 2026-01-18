from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def _helmert_contrast(self, levels):
    n = len(levels)
    contr = np.zeros((n, n - 1))
    contr[1:][np.diag_indices(n - 1)] = np.arange(1, n)
    contr[np.triu_indices(n - 1)] = -1
    return contr