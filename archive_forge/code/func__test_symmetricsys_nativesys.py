from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_symmetricsys_nativesys(NativeSys, nsteps=800, forgive=150):
    logexp = (sp.log, sp.exp)
    first_step = 0.0001
    rtol = atol = 1e-07
    k = [7.0, 3, 2]

    class TransformedNativeSys(TransformedSys, NativeSys):
        pass
    SS = symmetricsys(logexp, logexp, SuperClass=TransformedNativeSys)
    ts = SS.from_callback(decay_rhs, len(k) + 1, len(k))
    y0 = [1e-20] * (len(k) + 1)
    y0[0] = 1
    xout, yout, info = ts.integrate([1e-12, 1], y0, k, atol=atol, rtol=rtol, first_step=first_step, nsteps=nsteps)
    ref = np.array(bateman_full(y0, k + [0], xout - xout[0], exp=np.exp)).T
    np.set_printoptions(linewidth=240)
    assert np.allclose(yout, ref, rtol=rtol * forgive, atol=atol * forgive)