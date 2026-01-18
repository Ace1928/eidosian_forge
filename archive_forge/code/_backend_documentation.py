import typing
import types
import inspect
import functools
from . import _uarray
import copyreg
import pickle
import contextlib
from ._uarray import (  # type: ignore
Set a backend supporting all ``dispatchables``

    This is useful for functions that call multimethods without any dispatchable
    arguments. You can use :func:`determine_backend_multi` to ensure the same
    backend is used everywhere in a block of multimethod calls involving
    multiple arrays.

    Parameters
    ----------
    dispatchables: Sequence[Union[uarray.Dispatchable, Any]]
        The dispatchables that must be supported
    domain: string
        The domain to query for backends and set.
    coerce: bool
        Whether or not to allow coercion to the backend's types. Implies ``only``.
    only: bool
        Whether or not this should be the last backend to try.
    dispatch_type: Optional[Any]
        The default dispatch type associated with ``dispatchables``, aka
        ":ref:`marking <MarkingGlossary>`".

    See Also
    --------
    determine_backend: For a single dispatch value
    set_backend: For when you know which backend to set

    Notes
    -----

    Support is determined by the ``__ua_convert__`` protocol. Backends not
    supporting the type must return ``NotImplemented`` from their
    ``__ua_convert__`` if they don't support input of that type.

    Examples
    --------

    :func:`determine_backend` allows the backend to be set from a single
    object. :func:`determine_backend_multi` allows multiple objects to be
    checked simultaneously for support in the backend. Suppose we have a
    ``BackendAB`` which supports ``TypeA`` and ``TypeB`` in the same call,
    and a ``BackendBC`` that doesn't support ``TypeA``.

    >>> with ua.set_backend(ex.BackendAB), ua.set_backend(ex.BackendBC):
    ...     a, b = ex.TypeA(), ex.TypeB()
    ...     with ua.determine_backend_multi(
    ...         [ua.Dispatchable(a, "mark"), ua.Dispatchable(b, "mark")],
    ...         domain="ua_examples"
    ...     ):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, a, b)
    TypeA

    This won't call ``BackendBC`` because it doesn't support ``TypeA``.

    We can also use leave out the ``ua.Dispatchable`` if we specify the
    default ``dispatch_type`` for the ``dispatchables`` argument.

    >>> with ua.set_backend(ex.BackendAB), ua.set_backend(ex.BackendBC):
    ...     a, b = ex.TypeA(), ex.TypeB()
    ...     with ua.determine_backend_multi(
    ...         [a, b], dispatch_type="mark", domain="ua_examples"
    ...     ):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, a, b)
    TypeA

    