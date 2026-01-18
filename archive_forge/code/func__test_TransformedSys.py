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
def _test_TransformedSys(dep_tr, indep_tr, rtol, atol, first_step, forgive=1, y_zero=1e-20, t_zero=1e-12, **kwargs):
    k = [7.0, 3, 2]
    ts = symmetricsys(dep_tr, indep_tr).from_callback(decay_rhs, len(k) + 1, len(k))
    y0 = [y_zero] * (len(k) + 1)
    y0[0] = 1
    xout, yout, info = ts.integrate([t_zero, 1], y0, k, integrator='cvode', atol=atol, rtol=rtol, first_step=first_step, **kwargs)
    ref = np.array(bateman_full(y0, k + [0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, rtol=rtol * forgive, atol=atol * forgive)