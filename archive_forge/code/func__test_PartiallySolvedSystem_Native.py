from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_PartiallySolvedSystem_Native(NativeSys, integrator):

    class TransformedNativeSys(TransformedSys, NativeSys):
        pass
    logexp = get_logexp(1, 1e-24)
    NativeLogLogSys = symmetricsys(logexp, logexp, SuperClass=TransformedNativeSys)
    odesys = _get_decay3(lower_bounds=[0, 0, 0], linear_invariants=[[1, 1, 1]])
    n_sys = NativeSys.from_other(odesys)
    assert odesys.linear_invariants == n_sys.linear_invariants
    scaledsys = ScaledSys.from_other(odesys, dep_scaling=42)
    ns_sys = NativeSys.from_other(scaledsys)
    partsys = PartiallySolvedSystem.from_linear_invariants(scaledsys)
    np_sys = NativeSys.from_other(partsys)
    LogLogSys = symmetricsys(logexp, logexp)
    ll_scaledsys = LogLogSys.from_other(scaledsys)
    ll_partsys = LogLogSys.from_other(partsys)
    nll_scaledsys = NativeLogLogSys.from_other(scaledsys)
    nll_partsys = NativeLogLogSys.from_other(partsys)
    assert len(ll_scaledsys.nonlinear_invariants) > 0
    assert ll_scaledsys.nonlinear_invariants == nll_scaledsys.nonlinear_invariants
    y0 = [3.3, 2.4, 1.5]
    k = [3.5, 2.5, 0]
    systems = [odesys, n_sys, scaledsys, ns_sys, partsys, np_sys, ll_scaledsys, ll_partsys, nll_scaledsys, nll_partsys]
    for idx, system in enumerate(systems):
        result = system.integrate([0, 0.3, 0.5, 0.7, 0.9, 1.3], y0, k, integrator=integrator, atol=1e-09, rtol=1e-09)
        ref = np.array(bateman_full(y0, k, result.xout - result.xout[0], exp=np.exp)).T
        assert result.info['success']
        assert np.allclose(result.yout, ref)