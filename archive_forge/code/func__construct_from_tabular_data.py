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
@_define_handler(_handlers, False, False, False, False, True)
def _construct_from_tabular_data(self, obj, parent, nonlinear_function):
    idx = obj._index
    tabular_data = self._tabular_data
    if tabular_data is None:
        tabular_data = self._tabular_data_rule(parent, idx)
    points = [pt for pt in tabular_data.keys()]
    dimension = self._get_dimension_from_points(points)
    if dimension == 1:
        self._construct_one_dimensional_simplices_from_points(obj, points)
        return self._construct_from_univariate_function_and_segments(obj, _tabular_data_functor(tabular_data, tupleize=True))
    self._construct_simplices_from_multivariate_points(obj, points, dimension)
    return self._construct_from_function_and_simplices(obj, parent, _tabular_data_functor(tabular_data))