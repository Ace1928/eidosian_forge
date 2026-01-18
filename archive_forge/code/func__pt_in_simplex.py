import logging
from pyomo.common import DeveloperError
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.common.dependencies.scipy import spatial
from pyomo.contrib.piecewise.piecewise_linear_expression import (
from pyomo.core import Any, NonNegativeIntegers, value, Var
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.expression import Expression
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import UnindexedComponent_set
from pyomo.core.base.initializer import Initializer
import pyomo.core.expr as EXPR
def _pt_in_simplex(self, pt, simplex):
    dim = len(pt)
    if dim == 1:
        return self._points[simplex[0]][0] <= pt[0] and self._points[simplex[1]][0] >= pt[0]
    A = np.ones((dim + 1, dim + 1))
    b = np.array([x for x in pt] + [1])
    for j, extreme_point in enumerate(simplex):
        for i, coord in enumerate(self._points[extreme_point]):
            A[i, j] = coord
    if np.linalg.det(A) == 0:
        return False
    else:
        lambdas = np.linalg.solve(A, b)
    for l in lambdas:
        if l < -ZERO_TOLERANCE:
            return False
    return True