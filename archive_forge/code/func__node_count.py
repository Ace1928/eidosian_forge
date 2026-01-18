from collections import defaultdict
from .sympify import sympify, SympifyError
from sympy.utilities.iterables import iterable, uniq
def _node_count(e):
    if e.is_Float:
        return 0.5
    return 1 + sum(map(_node_count, e.args))