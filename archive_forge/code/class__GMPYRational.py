import builtins
from sympy.external.gmpy import HAS_GMPY, factorial, sqrt
from .pythonrational import PythonRational
from sympy.core.numbers import (
from sympy.core.numbers import (Float as SymPyReal, Integer as SymPyInteger, Rational as SymPyRational)
class _GMPYRational:

    def __init__(self, obj):
        pass