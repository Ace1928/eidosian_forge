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
def _to_func_decl_array(args):
    sz = len(args)
    _args = (FuncDecl * sz)()
    for i in range(sz):
        _args[i] = args[i].as_func_decl()
    return (_args, sz)