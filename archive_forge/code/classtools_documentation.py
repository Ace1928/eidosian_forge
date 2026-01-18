import functools
import inspect
import types
from typing import Type, Callable
Wrap the functionality the instance- or class method ``cls.name`` with additional behaviour
    ``before`` and ``after``.

    This mutates ``cls``, replacing the attribute ``name`` with the new functionality.  This is
    useful when creating class decorators.  The method is allowed to be defined on any parent class
    instead.

    If either ``before`` or ``after`` are given, they should be callables with a compatible
    signature to the method referred to.  They will be called immediately before or after the method
    as appropriate, and any return value will be ignored.

    Args:
        cls: the class to modify.
        name: the name of the method on the class to wrap.
        before: a callable that should be called before the method that is being wrapped.
        after: a callable that should be called after the method that is being wrapped.

    Raises:
        ValueError: if the named method is not defined on the class or any parent class.
    