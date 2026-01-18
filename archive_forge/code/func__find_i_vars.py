from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence
@staticmethod
def _find_i_vars(functions, intervals):
    i_vars = []
    for i in intervals:
        if i.v is None:
            continue
        elif i.v in i_vars:
            raise ValueError('Multiple intervals given for %s.' % str(i.v))
        i_vars.append(i.v)
    for f in functions:
        for a in f.free_symbols:
            if a not in i_vars:
                i_vars.append(a)
    return i_vars