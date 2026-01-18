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
def _test_mpmath():
    import sympy as sp
    x = sp.Symbol('x')
    symsys = SymbolicSys([(x, sp.exp(x))])
    tout = [0, 1e-09, 1e-07, 1e-05, 0.001, 0.1]
    xout, yout, info = symsys.integrate(tout, [1], integrator='mpmath')
    e = math.e
    ref = -math.log(1 / e - 0.1)
    assert abs(yout[-1, 0] - ref) < 4e-08