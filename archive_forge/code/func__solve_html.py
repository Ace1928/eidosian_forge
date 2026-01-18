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
def _solve_html(*args, **keywords):
    """Version of function `solve` that renders HTML output."""
    show = keywords.pop('show', False)
    s = Solver()
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