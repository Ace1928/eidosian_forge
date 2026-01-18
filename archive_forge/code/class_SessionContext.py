from __future__ import annotations
import logging # isort:skip
from abc import ABCMeta, abstractmethod
from typing import (
from ..core.types import ID
from ..document import Document
from ..settings import settings
class SessionContext(metaclass=ABCMeta):
    """ A harness for server-specific information and tasks related to
    Bokeh sessions.

    *This base class is probably not of interest to general users.*

    """
    _server_context: ServerContext
    _id: ID

    def __init__(self, server_context: ServerContext, session_id: ID) -> None:
        """

        """
        self._server_context = server_context
        self._id = session_id

    @property
    @abstractmethod
    def destroyed(self) -> bool:
        """ If ``True``, the session has been discarded and cannot be used.

        A new session with the same ID could be created later but this instance
        will not come back to life.

        """
        pass

    @property
    def id(self) -> ID:
        """ The unique ID for the session associated with this context.

        """
        return self._id

    @property
    def server_context(self) -> ServerContext:
        """ The server context for this session context

        """
        return self._server_context

    @abstractmethod
    def with_locked_document(self, func: Callable[[Document], Awaitable[None]]) -> Awaitable[None]:
        """ Runs a function with the document lock held, passing the
        document to the function.

        *Subclasses must implement this method.*

        Args:
            func (callable): function that takes a single parameter (the Document)
                and returns ``None`` or a ``Future``

        Returns:
            a ``Future`` containing the result of the function

        """
        pass