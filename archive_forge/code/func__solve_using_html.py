from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def _solve_using_html(s, *args, **keywords):
    """Version of function `solve_using` that renders HTML."""
    show = keywords.pop('show', False)
    if z3_debug():
        _z3_assert(isinstance(s, Solver), 'Solver object expected')
    s.set(**keywords)
    s.add(*args)
    if show:
        print('<b>Problem:</b>')
        print(s)
    r = s.check()
    if r == unsat:
        print('<b>no solution</b>')
    elif r == unknown:
        print('<b>failed to solve</b>')
        try:
            print(s.model())
        except Z3Exception:
            return
    else:
        if show:
            print('<b>Solution:</b>')
        print(s.model())