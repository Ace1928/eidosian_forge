from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_NativeSys__get_dx_max_source_code(NativeSys, forgive=20, **kwargs):
    dec3 = _get_decay3()
    odesys = NativeSys.from_other(dec3, namespace_override={'p_get_dx_max': 'AnyODE::ignore(y); return (1.0e-4 * x + 1.0e-3);'})
    y0, k = ([0.7, 0, 0], [7.0, 2, 3.0])
    xout, yout, info = odesys.integrate(1, y0, k, integrator='native', get_dx_max_factor=1.0, **kwargs)
    ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
    allclose_kw = dict(atol=kwargs['atol'] * forgive, rtol=kwargs['rtol'] * forgive)
    assert np.allclose(yout, ref, **allclose_kw)
    assert info['success']
    assert info['nfev'] > 10
    if 'n_steps' in info:
        print(info['n_steps'])
        assert 750 < info['n_steps'] <= 1000