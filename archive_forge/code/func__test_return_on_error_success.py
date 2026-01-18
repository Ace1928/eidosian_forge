from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_return_on_error_success(NativeSys):
    k, y0 = ([4, 3], (5, 4, 2))
    native = NativeSys.from_callback(decay_rhs, len(k) + 1, len(k), namespace_override={'p_rhs': '\n        f[0] = -m_p[0]*y[0];\n        f[1] = m_p[0]*y[0] - m_p[1]*y[1];\n        f[2] = m_p[1]*y[1];\n        if (x > 0.5) return AnyODE::Status::recoverable_error;\n        this->nfev++;\n        return AnyODE::Status::success;\n'})
    xout = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    result = native.integrate(xout, y0, k, atol=1e-11, rtol=1e-11, return_on_error=True, dx_max=0.05)
    nreached = result.info['nreached']
    assert nreached == 3
    ref = np.array(bateman_full(y0, k + [0], result.xout[:nreached] - xout[0], exp=np.exp)).T
    assert result.info['success'] is False
    assert np.allclose(result.yout[:nreached, :], ref, rtol=1e-08, atol=1e-08)