from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def _test_powell(sys_solver_pairs, x0=(1, 1), par=(1000.0,)):
    for sys, solver in sys_solver_pairs:
        x0, info = sys.solve(x0, par, solver=solver, tol=1e-12)
    assert info['success']
    x = sorted(x0)
    assert abs(_powell_ref[0] - x[0]) < 2e-11
    assert abs(_powell_ref[1] - x[1]) < 6e-10