from contextlib import contextmanager
from threading import local
from sympy.core.function import expand_mul
def _dotprodsimp(expr, withsimp=False):
    """Wrapper for simplify.dotprodsimp to avoid circular imports."""
    from sympy.simplify.simplify import dotprodsimp as dps
    return dps(expr, withsimp=withsimp)