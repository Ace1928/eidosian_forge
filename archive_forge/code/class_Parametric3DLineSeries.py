from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import arity, Function
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.printing.latex import latex
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .experimental_lambdify import (vectorized_lambdify, lambdify)
from sympy.plotting.textplot import textplot
class Parametric3DLineSeries(Line3DBaseSeries):
    """Representation for a 3D line consisting of three parametric SymPy
    expressions and a range."""
    is_parametric = True

    def __init__(self, expr_x, expr_y, expr_z, var_start_end, **kwargs):
        super().__init__()
        self.expr_x = sympify(expr_x)
        self.expr_y = sympify(expr_y)
        self.expr_z = sympify(expr_z)
        self.label = kwargs.get('label', None) or Tuple(self.expr_x, self.expr_y)
        self.var = sympify(var_start_end[0])
        self.start = float(var_start_end[1])
        self.end = float(var_start_end[2])
        self.nb_of_points = kwargs.get('nb_of_points', 300)
        self.line_color = kwargs.get('line_color', None)
        self._xlim = None
        self._ylim = None
        self._zlim = None

    def __str__(self):
        return '3D parametric cartesian line: (%s, %s, %s) for %s over %s' % (str(self.expr_x), str(self.expr_y), str(self.expr_z), str(self.var), str((self.start, self.end)))

    def get_parameter_points(self):
        np = import_module('numpy')
        return np.linspace(self.start, self.end, num=self.nb_of_points)

    def get_points(self):
        np = import_module('numpy')
        param = self.get_parameter_points()
        fx = vectorized_lambdify([self.var], self.expr_x)
        fy = vectorized_lambdify([self.var], self.expr_y)
        fz = vectorized_lambdify([self.var], self.expr_z)
        list_x = fx(param)
        list_y = fy(param)
        list_z = fz(param)
        list_x = np.array(list_x, dtype=np.float64)
        list_y = np.array(list_y, dtype=np.float64)
        list_z = np.array(list_z, dtype=np.float64)
        list_x = np.ma.masked_invalid(list_x)
        list_y = np.ma.masked_invalid(list_y)
        list_z = np.ma.masked_invalid(list_z)
        self._xlim = (np.amin(list_x), np.amax(list_x))
        self._ylim = (np.amin(list_y), np.amax(list_y))
        self._zlim = (np.amin(list_z), np.amax(list_z))
        return (list_x, list_y, list_z)