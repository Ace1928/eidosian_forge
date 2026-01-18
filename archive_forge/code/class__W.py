import contextlib
import itertools
import re
import typing
from enum import Enum
from typing import Callable
import sympy
from sympy import Add, Implies, sqrt
from sympy.core import Mul, Pow
from sympy.core import (S, pi, symbols, Function, Rational, Integer,
from sympy.functions import Piecewise, exp, sin, cos
from sympy.printing.smtlib import smtlib_code
from sympy.testing.pytest import raises, Failed
class _W(Enum):
    DEFAULTING_TO_FLOAT = re.compile('Could not infer type of `.+`. Defaulting to float.', re.I)
    WILL_NOT_DECLARE = re.compile('Non-Symbol/Function `.+` will not be declared.', re.I)
    WILL_NOT_ASSERT = re.compile('Non-Boolean expression `.+` will not be asserted. Converting to SMTLib verbatim.', re.I)