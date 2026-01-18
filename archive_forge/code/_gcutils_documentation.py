import weakref
import gc
from contextlib import contextmanager
from platform import python_implementation
Context manager to check that object is deallocated

    This is useful for checking that an object can be freed directly by
    reference counting, without requiring gc to break reference cycles.
    GC is disabled inside the context manager.

    This check is not available on PyPy.

    Parameters
    ----------
    func : callable
        Callable to create object to check
    \*args : sequence
        positional arguments to `func` in order to create object to check
    \*\*kwargs : dict
        keyword arguments to `func` in order to create object to check

    Examples
    --------
    >>> class C: pass
    >>> with assert_deallocated(C) as c:
    ...     # do something
    ...     del c

    >>> class C:
    ...     def __init__(self):
    ...         self._circular = self # Make circular reference
    >>> with assert_deallocated(C) as c: #doctest: +IGNORE_EXCEPTION_DETAIL
    ...     # do something
    ...     del c
    Traceback (most recent call last):
        ...
    ReferenceError: Remaining reference(s) to object
    