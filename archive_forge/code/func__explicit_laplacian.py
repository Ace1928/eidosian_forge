import pytest
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy import sparse
from scipy.sparse import csgraph
from scipy._lib._util import np_long, np_ulong
def _explicit_laplacian(x, normed=False):
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x)
    y = -1.0 * x
    for j in range(y.shape[0]):
        y[j, j] = x[j, j + 1:].sum() + x[j, :j].sum()
    if normed:
        d = np.diag(y).copy()
        d[d == 0] = 1.0
        y /= d[:, None] ** 0.5
        y /= d[None, :] ** 0.5
    return y