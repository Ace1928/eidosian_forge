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
@_define_handler(_handlers, True, True, False, False, False)
def _construct_from_function_and_points(self, obj, parent, nonlinear_function):
    idx = obj._index
    points = self._points_rule(parent, idx)
    dimension = self._get_dimension_from_points(points)
    if dimension == 1:
        self._construct_one_dimensional_simplices_from_points(obj, points)
        return self._construct_from_univariate_function_and_segments(obj, nonlinear_function)
    self._construct_simplices_from_multivariate_points(obj, points, dimension)
    return self._construct_from_function_and_simplices(obj, parent, nonlinear_function, simplices_are_user_defined=False)