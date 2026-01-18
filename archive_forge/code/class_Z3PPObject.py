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
class Z3PPObject:
    """Superclass for all Z3 objects that have support for pretty printing."""

    def use_pp(self):
        return True

    def _repr_html_(self):
        in_html = in_html_mode()
        set_html_mode(True)
        res = repr(self)
        set_html_mode(in_html)
        return res