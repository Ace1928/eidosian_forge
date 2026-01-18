from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sympify import sympify
class PreIncrement(Basic):
    """ Represents the pre-increment operator """
    nargs = 1