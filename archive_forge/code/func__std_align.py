from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from types import FunctionType
def _std_align(a):
    a = a.strip().lower()
    if len(a) > 1:
        return {'left': 'l', 'right': 'r', 'center': 'c'}.get(a, a)
    else:
        return {'<': 'l', '>': 'r', '^': 'c'}.get(a, a)