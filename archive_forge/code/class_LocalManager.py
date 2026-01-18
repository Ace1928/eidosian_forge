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
class LocalManager:
    """Manage releasing the data for the current context in one or more
    :class:`Local` and :class:`LocalStack` objects.

    This should not be needed for modern use cases, and may be removed
    in the future.

    :param locals: A local or list of locals to manage.

    .. versionchanged:: 2.1
        The ``ident_func`` was removed.

    .. versionchanged:: 0.7
        The ``ident_func`` parameter was added.

    .. versionchanged:: 0.6.1
        The :func:`release_local` function can be used instead of a
        manager.
    """
    __slots__ = ('locals',)

    def __init__(self, locals: None | (Local | LocalStack | t.Iterable[Local | LocalStack])=None) -> None:
        if locals is None:
            self.locals = []
        elif isinstance(locals, Local):
            self.locals = [locals]
        else:
            self.locals = list(locals)

    def cleanup(self) -> None:
        """Release the data in the locals for this context. Call this at
        the end of each request or use :meth:`make_middleware`.
        """
        for local in self.locals:
            release_local(local)

    def make_middleware(self, app: WSGIApplication) -> WSGIApplication:
        """Wrap a WSGI application so that local data is released
        automatically after the response has been sent for a request.
        """

        def application(environ: WSGIEnvironment, start_response: StartResponse) -> t.Iterable[bytes]:
            return ClosingIterator(app(environ, start_response), self.cleanup)
        return application

    def middleware(self, func: WSGIApplication) -> WSGIApplication:
        """Like :meth:`make_middleware` but used as a decorator on the
        WSGI application function.

        .. code-block:: python

            @manager.middleware
            def application(environ, start_response):
                ...
        """
        return update_wrapper(self.make_middleware(func), func)

    def __repr__(self) -> str:
        return f'<{type(self).__name__} storages: {len(self.locals)}>'