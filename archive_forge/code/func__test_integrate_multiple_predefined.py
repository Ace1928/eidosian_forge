from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
def _test_integrate_multiple_predefined(odes, **kwargs):
    _xout = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    _y0 = np.array([[1, 2], [2, 3], [3, 4]])
    _params = np.array([[5], [6], [7]])
    results = odes.integrate(_xout, _y0, params=_params, **kwargs)
    for idx in range(3):
        xout, yout, info = results[idx]
        ref = _y0[idx, 0] * np.exp(-_params[idx, 0] * xout)
        assert np.allclose(yout[:, 0], ref)
        assert np.allclose(yout[:, 1], _y0[idx, 0] - ref + _y0[idx, 1])
        assert info['nfev'] > 0
    return results