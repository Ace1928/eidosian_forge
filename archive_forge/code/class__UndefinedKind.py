from collections import defaultdict
from .cache import cacheit
from sympy.multipledispatch.dispatcher import (Dispatcher,
class _UndefinedKind(Kind):
    """
    Default kind for all SymPy object. If the kind is not defined for
    the object, or if the object cannot infer the kind from its
    arguments, this will be returned.

    Examples
    ========

    >>> from sympy import Expr
    >>> Expr().kind
    UndefinedKind
    """

    def __new__(cls):
        return super().__new__(cls)

    def __repr__(self):
        return 'UndefinedKind'