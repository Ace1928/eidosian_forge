from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_NativeSys__first_step_cb(NativeSys, forgive=20):
    dec3 = _get_decay3()
    dec3.first_step_expr = dec3.dep[0] * 1e-30
    odesys = NativeSys.from_other(dec3)
    y0, k = ([0.7, 0, 0], [1e+23, 2, 3.0])
    kwargs = dict(atol=1e-08, rtol=1e-08)
    xout, yout, info = odesys.integrate(5, y0, k, integrator='native', **kwargs)
    ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
    allclose_kw = dict(atol=kwargs['atol'] * forgive, rtol=kwargs['rtol'] * forgive)
    assert info['success'] and info['nfev'] > 10 and (info['nfev'] > 1) and (info['time_cpu'] < 100)
    assert np.allclose(yout, ref, **allclose_kw)