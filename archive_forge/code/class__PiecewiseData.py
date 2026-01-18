import logging
import math
import itertools
import operator
import types
import enum
from pyomo.common.log import is_debug_set
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.block import Block, _BlockData
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.var import Var, _VarData, IndexedVar
from pyomo.core.base.set_types import PositiveReals, NonNegativeReals, Binary
from pyomo.core.base.util import flatten_tuple
class _PiecewiseData(_BlockData):
    """
    This class defines the base class for all linearization
    and piecewise constraint generators..
    """

    def __init__(self, parent):
        _BlockData.__init__(self, parent)
        self._constructed = True
        self._bound_type = None
        self._domain_pts = None
        self._range_pts = None
        self._x = None
        self._y = None

    def updateBoundType(self, bound_type):
        self._bound_type = bound_type

    def updatePoints(self, domain_pts, range_pts):
        if not _isNonDecreasing(domain_pts):
            msg = "'%s' does not have a list of domain points that is non-decreasing"
            raise ValueError(msg % (self.name,))
        self._domain_pts = domain_pts
        self._range_pts = range_pts

    def build_constraints(self, functor, x_var, y_var):
        functor.construct(self, x_var, y_var)
        self.__dict__['_x'] = x_var
        self.__dict__['_y'] = y_var

    def referenced_variables(self):
        return (self._x, self._y)

    def __call__(self, x):
        if self._constructed is False:
            raise ValueError('Piecewise component %s has not been constructed yet' % self.name)
        for i in range(len(self._domain_pts) - 1):
            xL = self._domain_pts[i]
            xU = self._domain_pts[i + 1]
            if xL <= x and x <= xU:
                yL = self._range_pts[i]
                yU = self._range_pts[i + 1]
                if xL == xU:
                    return yU
                return yL + (yU - yL) / (xU - xL) * (x - xL)
        raise ValueError('The point %s is outside the list of domain points for Piecewise component %s. The valid point range is [%s,%s].' % (x, self.name, min(self._domain_pts), max(self._domain_pts)))