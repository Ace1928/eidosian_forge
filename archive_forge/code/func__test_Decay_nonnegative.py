from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_Decay_nonnegative(NativeSys):
    odesys = NativeSys.from_other(_get_decay3(lower_bounds=[0] * 3))
    y0, k = ([3.0, 2.0, 1.0], [3.5, 2.5, 0])
    xout, yout, info = odesys.integrate([1e-10, 1], y0, k, integrator='native')
    ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
    assert info['success'] and info['nfev'] > 10 and (info['nfev'] > 1) and (info['time_cpu'] < 100)
    assert np.allclose(yout, ref) and np.allclose(np.sum(yout, axis=1), sum(y0))