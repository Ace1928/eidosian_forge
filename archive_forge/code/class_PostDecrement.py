from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sympify import sympify
class PostDecrement(Basic):
    """ Represents the post-decrement operator """
    nargs = 1