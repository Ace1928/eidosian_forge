from .plot import BaseSeries, Plot
from .experimental_lambdify import experimental_lambdify, vectorized_lambdify
from .intervalmath import interval
from sympy.core.relational import (Equality, GreaterThan, LessThan,
from sympy.core.containers import Tuple
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.logic.boolalg import BooleanFunction
from sympy.polys.polyutils import _sort_gens
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import flatten
import warnings
def _get_meshes_grid(self):
    """Generates the mesh for generating a contour.

        In the case of equality, ``contour`` function of matplotlib can
        be used. In other cases, matplotlib's ``contourf`` is used.
        """
    equal = False
    if isinstance(self.expr, Equality):
        expr = self.expr.lhs - self.expr.rhs
        equal = True
    elif isinstance(self.expr, (GreaterThan, StrictGreaterThan)):
        expr = self.expr.lhs - self.expr.rhs
    elif isinstance(self.expr, (LessThan, StrictLessThan)):
        expr = self.expr.rhs - self.expr.lhs
    else:
        raise NotImplementedError('The expression is not supported for plotting in uniform meshed plot.')
    np = import_module('numpy')
    xarray = np.linspace(self.start_x, self.end_x, self.nb_of_points)
    yarray = np.linspace(self.start_y, self.end_y, self.nb_of_points)
    x_grid, y_grid = np.meshgrid(xarray, yarray)
    func = vectorized_lambdify((self.var_x, self.var_y), expr)
    z_grid = func(x_grid, y_grid)
    z_grid[np.ma.where(z_grid < 0)] = -1
    z_grid[np.ma.where(z_grid > 0)] = 1
    if equal:
        return (xarray, yarray, z_grid, 'contour')
    else:
        return (xarray, yarray, z_grid, 'contourf')