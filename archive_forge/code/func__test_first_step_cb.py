from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
def _test_first_step_cb(integrator, atol=1e-08, rtol=1e-08, forgive=10):
    odesys = ODESys(decay, decay_jac, dfdx=decay_dfdt, first_step_cb=lambda x, y, p, backend=None: y[0] * 1e-30)
    _y0 = [0.7, 0]
    k = [1e+23]
    xout, yout, info = odesys.integrate(5, _y0, k, integrator=integrator, atol=atol, rtol=rtol)
    ref = _y0[0] * np.exp(-k[0] * xout[:])
    assert np.allclose(yout[:, 0], ref, atol=atol * forgive, rtol=rtol * forgive)
    assert np.allclose(yout[:, 1], _y0[0] - ref + _y0[1], atol=atol * forgive, rtol=rtol * forgive)
    assert info['nfev'] > 0