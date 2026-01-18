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
def _to_tactic(t, ctx=None):
    if isinstance(t, Tactic):
        return t
    else:
        return Tactic(t, ctx)