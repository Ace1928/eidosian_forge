from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
def _test_integrate_multiple_adaptive(odes, **kwargs):
    _xout = np.array([[0, 1], [1, 2], [1, 7]])
    Ak = [[2, 3], [3, 4], [4, 5]]
    _y0 = [[0.0, A * k] for A, k in Ak]
    _params = [[k] for A, k in Ak]

    def _ref(A, k, t):
        return [A * np.sin(k * (t - t[0])), A * np.cos(k * (t - t[0])) * k]
    results = odes.integrate(_xout, _y0, _params, **kwargs)
    for idx in range(3):
        xout, yout, info = results[idx]
        ref = _ref(Ak[idx][0], Ak[idx][1], xout)
        assert np.allclose(yout[:, 0], ref[0], atol=1e-05, rtol=1e-05)
        assert np.allclose(yout[:, 1], ref[1], atol=1e-05, rtol=1e-05)
        assert info['nfev'] > 0
    return results