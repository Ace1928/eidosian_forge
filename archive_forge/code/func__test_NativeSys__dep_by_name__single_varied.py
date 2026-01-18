from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_NativeSys__dep_by_name__single_varied(NativeSys):
    tend, kf, y0 = (2, [4, 3], {'a': (5, 3, 7, 9, 1, 6, 11), 'b': 4, 'c': 2})
    y = sp.symarray('y', len(kf) + 1)
    dydt = decay_dydt_factory(kf)
    f = dydt(0, y)
    odesys = NativeSys(zip(y, f), names='a b c'.split(), dep_by_name=True)
    results = odesys.integrate(tend, y0, integrator='native')
    for idx in range(len(y0['a'])):
        xout, yout, info = results[idx]
        assert info['success']
        assert xout.size == yout.shape[0] and yout.shape[1] == 3
        ref = np.array(bateman_full([y0[k][idx] if k == 'a' else y0[k] for k in odesys.names], kf + [0], xout - xout[0], exp=np.exp)).T
        assert np.allclose(yout, ref)