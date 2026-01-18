from collections import defaultdict
from .cache import cacheit
from sympy.multipledispatch.dispatcher import (Dispatcher,
class _BooleanKind(Kind):
    """
    Kind for boolean objects.

    SymPy's ``S.true``, ``S.false``, and built-in ``True`` and ``False``
    have this kind. Boolean number ``1`` and ``0`` are not relevant.

    Examples
    ========

    >>> from sympy import S, Q
    >>> S.true.kind
    BooleanKind
    >>> Q.even(3).kind
    BooleanKind
    """

    def __new__(cls):
        return super().__new__(cls)

    def __repr__(self):
        return 'BooleanKind'