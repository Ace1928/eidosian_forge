from typing import Optional
import inspect
import sys
import warnings
from functools import wraps
def _get_indent(docstring: str) -> int:
    """

    Example:
        >>> def f():
        ...     '''Docstring summary.'''
        >>> f.__doc__
        'Docstring summary.'
        >>> _get_indent(f.__doc__)
        0

        >>> def g(foo):
        ...     '''Docstring summary.
        ...
        ...     Args:
        ...         foo: Does bar.
        ...     '''
        >>> g.__doc__
        'Docstring summary.\\n\\n    Args:\\n        foo: Does bar.\\n    '
        >>> _get_indent(g.__doc__)
        4

        >>> class A:
        ...     def h():
        ...         '''Docstring summary.
        ...
        ...         Returns:
        ...             None.
        ...         '''
        >>> A.h.__doc__
        'Docstring summary.\\n\\n        Returns:\\n            None.\\n        '
        >>> _get_indent(A.h.__doc__)
        8
    """
    if not docstring:
        return 0
    non_empty_lines = list(filter(bool, docstring.splitlines()))
    if len(non_empty_lines) == 1:
        return 0
    return len(non_empty_lines[1]) - len(non_empty_lines[1].lstrip())