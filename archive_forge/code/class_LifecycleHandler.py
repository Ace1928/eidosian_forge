from __future__ import annotations
import logging # isort:skip
from typing import Any, Callable
from ...document import Document
from ..application import ServerContext, SessionContext
from .handler import Handler
class LifecycleHandler(Handler):
    """ Load a script which contains server lifecycle callbacks.

    """
    _on_server_loaded: Callable[[ServerContext], None]
    _on_server_unloaded: Callable[[ServerContext], None]
    _on_session_created: Callable[[SessionContext], None]
    _on_session_destroyed: Callable[[SessionContext], None]

    def __init__(self) -> None:
        super().__init__()
        self._on_server_loaded = _do_nothing
        self._on_server_unloaded = _do_nothing
        self._on_session_created = _do_nothing
        self._on_session_destroyed = _do_nothing

    @property
    def safe_to_fork(self) -> bool:
        return True

    def modify_document(self, doc: Document) -> None:
        """ This handler does not make any modifications to the Document.

        Args:
            doc (Document) : A Bokeh Document to update in-place

                *This handler does not modify the document*

        Returns:
            None

        """
        pass

    def on_server_loaded(self, server_context: ServerContext) -> None:
        """ Execute `on_server_unloaded`` from the configured module (if
        it is defined) when the server is first started.

        Args:
            server_context (ServerContext) :

        """
        return self._on_server_loaded(server_context)

    def on_server_unloaded(self, server_context: ServerContext) -> None:
        """ Execute ``on_server_unloaded`` from the configured module (if
        it is defined) when the server cleanly exits. (Before stopping the
        server's ``IOLoop``.)

        Args:
            server_context (ServerContext) :

        .. warning::
            In practice this code may not run, since servers are often killed
            by a signal.

        """
        return self._on_server_unloaded(server_context)

    async def on_session_created(self, session_context: SessionContext) -> None:
        """ Execute ``on_session_created`` from the configured module (if
        it is defined) when a new session is created.

        Args:
            session_context (SessionContext) :

        """
        return self._on_session_created(session_context)

    async def on_session_destroyed(self, session_context: SessionContext) -> None:
        """ Execute ``on_session_destroyed`` from the configured module (if
        it is defined) when a new session is destroyed.

        Args:
            session_context (SessionContext) :

        """
        return self._on_session_destroyed(session_context)