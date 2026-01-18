from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence
def _fill_i_vars(self, i_vars):
    self.i_vars = [Symbol(str(i)) for i in self.i_vars]
    for i in range(len(i_vars)):
        self.i_vars[i] = i_vars[i]