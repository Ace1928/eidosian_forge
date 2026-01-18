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
def add_cover(self, level, predicate, property):
    """Add property to predicate for the level'th unfolding.
        -1 is treated as infinity (infinity)
        """
    Z3_fixedpoint_add_cover(self.ctx.ref(), self.fixedpoint, level, predicate.ast, property.ast)