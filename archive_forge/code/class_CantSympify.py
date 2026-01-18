from __future__ import annotations
from typing import Any, Callable
from inspect import getmro
import string
from sympy.core.random import choice
from .parameters import global_parameters
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable
from .basic import Basic
class CantSympify:
    """
    Mix in this trait to a class to disallow sympification of its instances.

    Examples
    ========

    >>> from sympy import sympify
    >>> from sympy.core.sympify import CantSympify

    >>> class Something(dict):
    ...     pass
    ...
    >>> sympify(Something())
    {}

    >>> class Something(dict, CantSympify):
    ...     pass
    ...
    >>> sympify(Something())
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: {}

    """
    __slots__ = ()