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
def _get_simplices_from_arg(self, simplices):
    self._simplices = []
    known_points = set()
    point_to_index = {}
    for simplex in simplices:
        extreme_pts = []
        for pt in simplex:
            if pt not in known_points:
                known_points.add(pt)
                if hasattr(pt, '__len__'):
                    self._points.append(pt)
                else:
                    self._points.append((pt,))
                point_to_index[pt] = len(self._points) - 1
            extreme_pts.append(point_to_index[pt])
        self._simplices.append(tuple(extreme_pts))