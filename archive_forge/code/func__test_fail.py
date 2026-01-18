from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def _test_fail(solver, **kwargs):

    def _f(x, p):
        return [p[0] + x[0] ** 2]

    def _j(x, p):
        return [[2 * x[0]]]
    ns = NeqSys(1, 1, _f, jac=_j)
    x, res = ns.solve([1], [1], solver=solver, **kwargs)
    assert len(x) == 1
    assert abs(x[0]) < 1e-08
    assert not res['success']