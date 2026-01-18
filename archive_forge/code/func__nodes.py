from collections import defaultdict
from .sympify import sympify, SympifyError
from sympy.utilities.iterables import iterable, uniq
def _nodes(e):
    """
    A helper for ordered() which returns the node count of ``e`` which
    for Basic objects is the number of Basic nodes in the expression tree
    but for other objects is 1 (unless the object is an iterable or dict
    for which the sum of nodes is returned).
    """
    from .basic import Basic
    from .function import Derivative
    if isinstance(e, Basic):
        if isinstance(e, Derivative):
            return _nodes(e.expr) + sum((i[1] if i[1].is_Number else _nodes(i[1]) for i in e.variable_count))
        return _node_count(e)
    elif iterable(e):
        return 1 + sum((_nodes(ei) for ei in e))
    elif isinstance(e, dict):
        return 1 + sum((_nodes(k) + _nodes(v) for k, v in e.items()))
    else:
        return 1