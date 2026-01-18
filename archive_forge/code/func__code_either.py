from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def _code_either(self, intercept, levels):
    n = len(levels)
    scores = self.scores
    if scores is None:
        scores = np.arange(n)
    scores = np.asarray(scores, dtype=float)
    if len(scores) != n:
        raise PatsyError('number of levels (%s) does not match number of scores (%s)' % (n, len(scores)))
    scores -= scores.mean()
    raw_poly = scores.reshape((-1, 1)) ** np.arange(n).reshape((1, -1))
    q, r = np.linalg.qr(raw_poly)
    q *= np.sign(np.diag(r))
    q /= np.sqrt(np.sum(q ** 2, axis=1))
    q[:, 0] = 1
    names = ['.Constant', '.Linear', '.Quadratic', '.Cubic']
    names += ['^%s' % (i,) for i in range(4, n)]
    names = names[:n]
    if intercept:
        return ContrastMatrix(q, names)
    else:
        return ContrastMatrix(q[:, 1:], names[1:])