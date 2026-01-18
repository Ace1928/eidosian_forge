from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_NativeSys__band(NativeSys):
    tend, k, y0 = (2, [4, 3], (5, 4, 2))
    y = sp.symarray('y', len(k) + 1)
    dydt = decay_dydt_factory(k)
    f = dydt(0, y)
    odesys = NativeSys(zip(y, f), band=(1, 0))
    xout, yout, info = odesys.integrate(tend, y0, integrator='native')
    ref = np.array(bateman_full(y0, k + [0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref)