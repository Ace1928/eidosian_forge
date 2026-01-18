from __future__ import print_function, absolute_import, division
from collections import defaultdict, OrderedDict
from itertools import product
import math
import numpy as np
import pytest
from .. import ODESys
from ..core import integrate_auto_switch, chained_parameter_variation
from ..symbolic import SymbolicSys, ScaledSys, symmetricsys, PartiallySolvedSystem, get_logexp, _group_invariants
from ..util import requires, pycvodes_double, pycvodes_klu
from .bateman import bateman_full  # analytic, never mind the details
from .test_core import vdp_f
from . import _cetsa
def _test_bateman(SymbSys, **kwargs):
    import sympy as sp
    tend, k, y0 = (2, [4, 3], (5, 4, 2))
    y = sp.symarray('y', len(k) + 1)
    dydt = decay_dydt_factory(k)
    f = dydt(0, y)
    odesys = SymbSys(zip(y, f), **kwargs)
    xout, yout, info = odesys.integrate(tend, y0, integrator='scipy')
    ref = np.array(bateman_full(y0, k + [0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref)