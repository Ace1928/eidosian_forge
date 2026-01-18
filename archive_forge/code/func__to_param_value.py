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
def _to_param_value(val):
    if isinstance(val, bool):
        return 'true' if val else 'false'
    return str(val)