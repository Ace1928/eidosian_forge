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
def _matplotlib_list(interval_list):
    """
    Returns lists for matplotlib ``fill`` command from a list of bounding
    rectangular intervals
    """
    xlist = []
    ylist = []
    if len(interval_list):
        for intervals in interval_list:
            intervalx = intervals[0]
            intervaly = intervals[1]
            xlist.extend([intervalx.start, intervalx.start, intervalx.end, intervalx.end, None])
            ylist.extend([intervaly.start, intervaly.end, intervaly.end, intervaly.start, None])
    else:
        xlist.extend((None, None, None, None))
        ylist.extend((None, None, None, None))
    return (xlist, ylist)