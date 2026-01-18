from .basic import Basic
from .sorting import ordered
from .sympify import sympify
from sympy.utilities.iterables import iterable
def _use(expr, level):
    if not level:
        return func(expr, *args, **kwargs)
    elif expr.is_Atom:
        return expr
    else:
        level -= 1
        _args = [_use(arg, level) for arg in expr.args]
        return expr.__class__(*_args)