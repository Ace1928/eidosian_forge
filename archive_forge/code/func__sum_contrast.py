from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def _sum_contrast(self, levels):
    n = len(levels)
    omit_i = self._omit_i(levels)
    eye = np.eye(n - 1)
    out = np.empty((n, n - 1))
    out[:omit_i, :] = eye[:omit_i, :]
    out[omit_i, :] = -1
    out[omit_i + 1:, :] = eye[omit_i:, :]
    return out