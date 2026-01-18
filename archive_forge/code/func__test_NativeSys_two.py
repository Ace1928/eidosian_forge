from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_NativeSys_two(NativeSys, nsteps=500):
    native1 = NativeSys.from_callback(vdp_f, 2, 1)
    tend2, k2, y02 = (2, [4, 3], (5, 4, 2))
    atol2, rtol2 = (1e-11, 1e-11)
    native2 = NativeSys.from_callback(decay_dydt_factory(k2), len(k2) + 1)
    xout1, yout1, info1 = native1.integrate([0, 1, 2], [1, 0], params=[2.0], nsteps=nsteps)
    xout2, yout2, info2 = native2.integrate(tend2, y02, atol=atol2, rtol=rtol2, nsteps=nsteps)
    ref1 = [[1, 0], [0.44449086, -1.32847148], [-1.89021896, -0.71633577]]
    assert np.allclose(yout1, ref1)
    if 'nfev' in info1:
        assert info1['nfev'] > 0
    ref2 = np.array(bateman_full(y02, k2 + [0], xout2 - xout2[0], exp=np.exp)).T
    assert np.allclose(yout2, ref2, rtol=150 * rtol2, atol=150 * atol2)