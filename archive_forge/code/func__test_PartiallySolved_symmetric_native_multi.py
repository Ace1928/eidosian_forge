from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_PartiallySolved_symmetric_native_multi(NativeSys, multiple=False, forgive=1, **kwargs):
    trnsfsys = _get_transformed_partially_solved_system(NativeSys, multiple)
    y0s = [[3.0, 2.0, 1.0], [3.1, 2.1, 1.1], [3.2, 2.3, 1.2], [3.6, 2.4, 1.3]]
    ks = [[3.5, 2.5, 0], [3.3, 2.4, 0], [3.2, 2.1, 0], [3.3, 2.4, 0]]
    results = trnsfsys.integrate([(1e-10, 1)] * len(ks), y0s, ks, integrator='native', **kwargs)
    allclose_kw = dict(atol=kwargs.get('atol', 1e-08) * forgive, rtol=kwargs.get('rtol', 1e-08) * forgive)
    for i, (y0, k) in enumerate(zip(y0s, ks)):
        xout, yout, info = results[i]
        ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
        assert info['success']
        assert info['nfev'] > 10
        assert info['nfev'] > 1
        assert info['time_cpu'] < 100
        assert np.allclose(yout, ref, **allclose_kw)
        assert np.allclose(np.sum(yout, axis=1), sum(y0), **allclose_kw)