import pytest
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy import sparse
from scipy.sparse import csgraph
from scipy._lib._util import np_long, np_ulong
def _assert_allclose_sparse(a, b, **kwargs):
    if sparse.issparse(a):
        a = a.toarray()
    if sparse.issparse(b):
        b = b.toarray()
    assert_allclose(a, b, **kwargs)