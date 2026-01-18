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
def _construct_simplices_from_multivariate_points(self, obj, points, dimension):
    try:
        triangulation = spatial.Delaunay(points)
    except (spatial.QhullError, ValueError) as error:
        logger.error('Unable to triangulate the set of input points.')
        raise
    obj._points = [pt for pt in map(tuple, triangulation.points)]
    obj._simplices = []
    for simplex in triangulation.simplices:
        points = triangulation.points[simplex].transpose()
        if np.linalg.matrix_rank(points[:, 1:] - np.append(points[:, :dimension - 1], points[:, [0]], axis=1)) == dimension:
            obj._simplices.append(tuple(sorted(simplex)))
    for pt in triangulation.coplanar:
        logger.info('The Delaunay triangulation dropped the point with index %s from the triangulation.' % pt[0])