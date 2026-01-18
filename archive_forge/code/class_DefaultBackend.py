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
class DefaultBackend(BaseBackend):

    def __new__(cls, parent):
        matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
        if matplotlib:
            return MatplotlibBackend(parent)
        else:
            return TextBackend(parent)