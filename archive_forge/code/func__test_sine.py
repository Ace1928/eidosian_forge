from __future__ import (absolute_import, division, print_function)
import numpy as np
from .. import ODESys
from ..util import requires
from .test_core import sine, sine_jac
def _test_sine(use_deriv, atol=1e-08, rtol=1e-08, forgive=10000.0):
    odesys = ODESys(sine, sine_jac)
    A, k = (2, 3)
    result = odesys.integrate(np.linspace(0, 1, 17), [0, A * k], [k], atol=atol, rtol=rtol)
    x_probe = 2 ** (-0.5)
    assert result.info['success']
    ref = lambda x: np.array([A * np.sin(k * x), A * np.cos(k * x) * k]).T
    est, est_err = result.at(x_probe, use_deriv=use_deriv)
    real_err = np.abs(est - ref(x_probe))
    assert np.all(real_err < est_err)
    assert np.allclose(ref(x_probe), est, atol=atol * forgive, rtol=rtol * forgive)
    probes = np.linspace(0, 1, 11)
    est2, est_err2 = result.at(probes, linear=True)
    assert est_err2 is None
    assert np.allclose(ref(probes), est2, atol=0.03)