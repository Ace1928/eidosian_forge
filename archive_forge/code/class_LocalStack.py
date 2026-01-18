from __future__ import annotations
import copy
import math
import operator
import typing as t
from contextvars import ContextVar
from functools import partial
from functools import update_wrapper
from operator import attrgetter
from .wsgi import ClosingIterator
class LocalStack(t.Generic[T]):
    """Create a stack of context-local data. This wraps a
    :class:`ContextVar` containing a :class:`list` value.

    This may incur a performance penalty compared to using individual
    context vars, as it has to copy data to avoid mutating the list
    between nested contexts.

    :param context_var: The :class:`~contextvars.ContextVar` to use as
        storage for this local. If not given, one will be created.
        Context vars not created at the global scope may interfere with
        garbage collection.

    .. versionchanged:: 2.0
        Uses ``ContextVar`` instead of a custom storage implementation.

    .. versionadded:: 0.6.1
    """
    __slots__ = ('_storage',)

    def __init__(self, context_var: ContextVar[list[T]] | None=None) -> None:
        if context_var is None:
            context_var = ContextVar(f'werkzeug.LocalStack<{id(self)}>.storage')
        self._storage = context_var

    def __release_local__(self) -> None:
        self._storage.set([])

    def push(self, obj: T) -> list[T]:
        """Add a new item to the top of the stack."""
        stack = self._storage.get([]).copy()
        stack.append(obj)
        self._storage.set(stack)
        return stack

    def pop(self) -> T | None:
        """Remove the top item from the stack and return it. If the
        stack is empty, return ``None``.
        """
        stack = self._storage.get([])
        if len(stack) == 0:
            return None
        rv = stack[-1]
        self._storage.set(stack[:-1])
        return rv

    @property
    def top(self) -> T | None:
        """The topmost item on the stack.  If the stack is empty,
        `None` is returned.
        """
        stack = self._storage.get([])
        if len(stack) == 0:
            return None
        return stack[-1]

    def __call__(self, name: str | None=None, *, unbound_message: str | None=None) -> LocalProxy:
        """Create a :class:`LocalProxy` that accesses the top of this
        local stack.

        :param name: If given, the proxy access this attribute of the
            top item, rather than the item itself.
        :param unbound_message: The error message that the proxy will
            show if the stack is empty.
        """
        return LocalProxy(self, name, unbound_message=unbound_message)