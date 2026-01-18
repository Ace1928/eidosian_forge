import math
from operator import add
from functools import reduce
import pytest
from chempy import Substance
from chempy.units import (
from ..testing import requires
from ..pyutil import defaultkeydict
from .._expr import (
from ..parsing import parsing_library
class Pressure1(Expr):
    argument_names = ('n',)
    parameter_keys = ('temperature', 'volume', 'R')

    def __call__(self, variables, backend=None):
        n, = self.all_args(variables, backend=backend)
        T, V, R = self.all_params(variables, backend=backend)
        return n * R * T / V