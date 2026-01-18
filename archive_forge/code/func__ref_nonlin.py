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
def _ref_nonlin(y0, k, t):
    X, Y, Z = (y0[2], max(y0[:2]), min(y0[:2]))
    kf, kb = k
    x0 = Y * kf
    x1 = Z * kf
    x2 = 2 * X * kf
    x3 = -kb - x0 - x1
    x4 = -x2 + x3
    x5 = np.sqrt(-4 * kf * (X ** 2 * kf + X * x0 + X * x1 + Z * x0) + x4 ** 2)
    x6 = kb + x0 + x1 + x5
    x7 = (x3 + x5) * np.exp(-t * x5)
    x8 = x3 - x5
    return (x4 * x8 + x5 * x8 + x7 * (x2 + x6)) / (2 * kf * (x6 + x7))