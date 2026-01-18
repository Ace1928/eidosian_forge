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
@_define_handler(_handlers, True, False, True, False, False)
def _construct_from_function_and_simplices(self, obj, parent, nonlinear_function, simplices_are_user_defined=True):
    if obj._simplices is None:
        obj._get_simplices_from_arg(self._simplices_rule(parent, obj._index))
    simplices = obj._simplices
    if len(simplices) < 1:
        raise ValueError('Cannot construct PiecewiseLinearFunction with empty list of simplices')
    dimension = len(simplices[0]) - 1
    if dimension == 1:
        return self._construct_from_univariate_function_and_segments(obj, nonlinear_function)
    A = np.ones((dimension + 2, dimension + 2))
    b = np.zeros(dimension + 2)
    b[-1] = 1
    for num_piece, simplex in enumerate(simplices):
        for i, pt_idx in enumerate(simplex):
            pt = obj._points[pt_idx]
            for j, val in enumerate(pt):
                A[i, j] = val
            A[i, j + 1] = nonlinear_function(*pt)
        A[i + 1, :] = 0
        A[i + 1, dimension] = -1
        try:
            normal = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as e:
            logger.warning('LinAlgError: %s' % e)
            msg = 'When calculating the hyperplane approximation over the simplex with index %s, the matrix was unexpectedly singular. This likely means that this simplex is degenerate' % num_piece
            if simplices_are_user_defined:
                raise ValueError(msg)
            raise DeveloperError(msg + ' and that it should have been filtered out of the triangulation')
        obj._linear_functions.append(_multivariate_linear_functor(normal))
    return obj