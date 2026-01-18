from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_render_native_code_cse(NativeSys):
    from pyodesys.symbolic import SymbolicSys
    from sympy import symbols, log, exp
    import numpy as np
    symbs = symbols('N U A L NL t T He_dis Se_dis Cp_dis Tref_dis He_u Se_u Cp_u Tref_u Ha_agg Sa_agg Ha_as Sa_as Ha_f Sa_f R h k_B')
    di = {s.name: s for s in symbs}

    class NS:
        pass
    ns = NS()
    ns.__dict__.update(di)

    def _gibbs(H, S, Cp, Tref):
        H2 = H + Cp * (ns.T - Tref)
        S2 = S + Cp * log(ns.T / Tref)
        return exp(-(H2 - ns.T * S2) / (ns.R * ns.T))

    def _eyring(H, S):
        return ns.k_B / ns.h * ns.T * exp(-(H - ns.T * S) / (ns.R * ns.T))
    k_agg = _eyring(di['Ha_agg'], di['Sa_agg'])
    k_as = _eyring(di['Ha_as'], di['Sa_as'])
    k_f = _eyring(di['Ha_f'], di['Sa_f'])
    k_dis = k_as * _gibbs(*[di[k] for k in ('He_dis', 'Se_dis', 'Cp_dis', 'Tref_dis')])
    k_u = k_f * _gibbs(*[di[k] for k in ('He_u', 'Se_u', 'Cp_u', 'Tref_u')])
    r_agg = k_agg * ns.U
    r_as = k_as * ns.N * ns.L
    r_f = k_f * ns.U
    r_dis = k_dis * ns.NL
    r_u = k_u * ns.N
    exprs = [-r_as + r_f + r_dis - r_u, -r_agg - r_f + r_u, r_agg, r_dis - r_as, r_as - r_dis]

    def _solve(odesys, **kwargs):
        default_c0 = defaultdict(float, {'N': 1e-09, 'L': 1e-08})
        params = dict(R=8.314472, k_B=1.3806504e-23, h=6.62606896e-34, He_dis=-45000.0, Se_dis=-400, Cp_dis=1780.0, Tref_dis=298.15, He_u=60000.0, Se_u=130.5683, Cp_u=20500.0, Tref_u=298.15, Ha_agg=106000.0, Sa_agg=70, Ha_as=4000.0, Sa_as=-10, Ha_f=90000.0, Sa_f=50, T=50 + 273.15)
        return odesys.integrate(3600 * 24, [default_c0[s.name] for s in symbs[:5]], [params[s.name] for s in symbs[6:]], **kwargs)
    symbolic = SymbolicSys(zip(symbs[:5], exprs), symbs[5], params=symbs[6:])
    kw = dict(integrator='cvode', nsteps=35000, atol=1e-11, rtol=1e-11)
    ref = _solve(symbolic, **kw)
    assert ref.info['success']
    native = NativeSys.from_other(symbolic)
    sol = _solve(native, **kw)
    assert sol.info['success']
    assert np.allclose(sol.yout[-1, :], ref.yout[-1, :])