import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def _check_calling_conventions(self, X, metric, eps=1e-07, **kwargs):
    try:
        y1 = pdist(X, metric=metric, **kwargs)
        y2 = pdist(X, metric=eval(metric), **kwargs)
        y3 = pdist(X, metric='test_' + metric, **kwargs)
    except Exception as e:
        e_cls = e.__class__
        if verbose > 2:
            print(e_cls.__name__)
            print(e)
        with pytest.raises(e_cls):
            pdist(X, metric=metric, **kwargs)
        with pytest.raises(e_cls):
            pdist(X, metric=eval(metric), **kwargs)
        with pytest.raises(e_cls):
            pdist(X, metric='test_' + metric, **kwargs)
    else:
        assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)
        assert_allclose(y1, y3, rtol=eps, verbose=verbose > 2)