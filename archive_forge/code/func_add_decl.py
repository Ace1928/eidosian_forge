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
def add_decl(self, decl):
    Z3_parser_context_add_decl(self.ctx.ref(), self.pctx, decl.as_ast())