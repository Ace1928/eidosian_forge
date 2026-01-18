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
class ScopedConstructor:
    """Auxiliary object used to create Z3 datatypes."""

    def __init__(self, c, ctx):
        self.c = c
        self.ctx = ctx

    def __del__(self):
        if self.ctx.ref() is not None and Z3_del_constructor is not None:
            Z3_del_constructor(self.ctx.ref(), self.c)