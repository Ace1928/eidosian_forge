import sys
import types
import inspect
from functools import wraps, update_wrapper
from sympy.utilities.exceptions import sympy_deprecation_warning
class no_attrs_in_subclass:
    """Don't 'inherit' certain attributes from a base class

    >>> from sympy.utilities.decorator import no_attrs_in_subclass

    >>> class A(object):
    ...     x = 'test'

    >>> A.x = no_attrs_in_subclass(A, A.x)

    >>> class B(A):
    ...     pass

    >>> hasattr(A, 'x')
    True
    >>> hasattr(B, 'x')
    False

    """

    def __init__(self, cls, f):
        self.cls = cls
        self.f = f

    def __get__(self, instance, owner=None):
        if owner == self.cls:
            if hasattr(self.f, '__get__'):
                return self.f.__get__(instance, owner)
            return self.f
        raise AttributeError