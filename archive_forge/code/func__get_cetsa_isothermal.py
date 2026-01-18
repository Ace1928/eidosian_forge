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
def _get_cetsa_isothermal():
    names = ('NL', 'N', 'L', 'U', 'A')
    k_names = ('dis', 'as', 'un', 'fo', 'ag')

    def i(n):
        return names.index(n)

    def k(n):
        return k_names.index(n)

    def rhs(x, y, p):
        r = {'diss': p['dis'] * y['NL'], 'asso': p['as'] * y['N'] * y['L'], 'unfo': p['un'] * y['N'], 'fold': p['fo'] * y['U'], 'aggr': p['ag'] * y['U']}
        return {'NL': r['asso'] - r['diss'], 'N': r['diss'] - r['asso'] + r['fold'] - r['unfo'], 'L': r['diss'] - r['asso'], 'U': r['unfo'] - r['fold'] - r['aggr'], 'A': r['aggr']}
    return SymbolicSys.from_callback(rhs, dep_by_name=True, par_by_name=True, names=names, param_names=k_names)