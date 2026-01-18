from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def _test_neqsys_solve_series(solver):
    ns = NeqSys(2, 2, f, jac=j)
    x, sol = ns.solve_series(solver, [0, 0], [0], var_data=[2, 3], var_idx=0)
    assert abs(x[0, 0] - 0.5) < 2e-07
    assert abs(x[0, 1] + 0.5) < 2e-07
    assert abs(x[1, 0] - 0.8411639) < 2e-07
    assert abs(x[1, 1] - 0.1588361) < 2e-07