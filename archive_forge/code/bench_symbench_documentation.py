from sympy.core.random import random
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import factor
from sympy.simplify.simplify import simplify
from sympy.abc import x, y, z
from timeit import default_timer as clock
e=(x+y+z+1)**7;f=e*(e+1);f.expand()