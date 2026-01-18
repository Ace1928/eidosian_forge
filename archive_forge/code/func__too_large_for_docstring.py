from __future__ import annotations
from typing import Any
import builtins
import inspect
import keyword
import textwrap
import linecache
from sympy.external import import_module # noqa:F401
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import (is_sequence, iterable,
from sympy.utilities.misc import filldedent
def _too_large_for_docstring(expr, limit):
    """Decide whether an ``Expr`` is too large to be fully rendered in a
    ``lambdify`` docstring.

    This is a fast alternative to ``count_ops``, which can become prohibitively
    slow for large expressions, because in this instance we only care whether
    ``limit`` is exceeded rather than counting the exact number of nodes in the
    expression.

    Parameters
    ==========
    expr : ``Expr``, (nested) ``list`` of ``Expr``, or ``Matrix``
        The same objects that can be passed to the ``expr`` argument of
        ``lambdify``.
    limit : ``int`` or ``None``
        The threshold above which an expression contains too many nodes to be
        usefully rendered in the docstring. If ``None`` then there is no limit.

    Returns
    =======
    bool
        ``True`` if the number of nodes in the expression exceeds the limit,
        ``False`` otherwise.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.lambdify import _too_large_for_docstring
    >>> expr = x
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    False
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    Does this split it?

    >>> expr = [x, y, z]
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> expr = [x, [y], z, [[x+y], [x*y*z, [x+y+z]]]]
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> expr = ((x + y + z)**5).expand()
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    True
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> from sympy import Matrix
    >>> expr = Matrix([[(x + y + z), ((x + y + z)**2).expand(),
    ...                 ((x + y + z)**3).expand(), ((x + y + z)**4).expand()]])
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 1000)
    False
    >>> _too_large_for_docstring(expr, 100)
    True
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    """
    from sympy.core.traversal import postorder_traversal
    if limit is None:
        return False
    i = 0
    for _ in postorder_traversal(expr):
        i += 1
        if i > limit:
            return True
    return False