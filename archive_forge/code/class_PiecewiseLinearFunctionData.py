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
class PiecewiseLinearFunctionData(_BlockData):
    _Block_reserved_words = Any

    def __init__(self, component=None):
        _BlockData.__init__(self, component)
        with self._declare_reserved_components():
            self._expressions = Expression(NonNegativeIntegers)
            self._transformed_exprs = ComponentMap()
            self._simplices = None
            self._points = []
            self._linear_functions = []

    def __call__(self, *args):
        """
        Returns a PiecewiseLinearExpression which is an instance of this
        function applied to the variables and/or constants specified in args.
        """
        if all((type(arg) in EXPR.native_types or not arg.is_potentially_variable() for arg in args)):
            return self._evaluate(*args)
        else:
            expr = PiecewiseLinearExpression(args, self)
            idx = id(expr)
            self._expressions[idx] = expr
            return self._expressions[idx]

    def _evaluate(self, *args):
        if self._simplices is None:
            raise RuntimeError('Cannot evaluate PiecewiseLinearFunction--it appears it is not fully defined. (No simplices are stored.)')
        pt = [value(arg) for arg in args]
        for simplex, func in zip(self._simplices, self._linear_functions):
            if self._pt_in_simplex(pt, simplex):
                return func(*args)
        raise ValueError("Unsuccessful evaluation of PiecewiseLinearFunction '%s' at point (%s). Is the point in the function's domain?" % (self.name, ', '.join((str(arg) for arg in args))))

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

    def map_transformation_var(self, pw_expr, v):
        """
        Records on the PiecewiseLinearFunction object that the transformed
        form of the PiecewiseLinearExpression object pw_expr is the Var v.
        """
        self._transformed_exprs[self._expressions[id(pw_expr)]] = v

    def get_transformation_var(self, pw_expr):
        """
        Returns the Var that replaced the PiecewiseLinearExpression 'pw_expr'
        after transformation, or None if 'pw_expr' has not been transformed.
        """
        if pw_expr in self._transformed_exprs:
            return self._transformed_exprs[pw_expr]
        else:
            return None