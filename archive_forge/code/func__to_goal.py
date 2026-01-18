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
def _to_goal(a):
    if isinstance(a, BoolRef):
        goal = Goal(ctx=a.ctx)
        goal.add(a)
        return goal
    else:
        return a