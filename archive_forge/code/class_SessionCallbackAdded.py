from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class SessionCallbackAdded(DocumentChangedEvent):
    """ A concrete event representing a change to add a new callback (e.g.
    periodic, timeout, or "next tick") to a Document.

    """

    def __init__(self, document: Document, callback: SessionCallback) -> None:
        """

        Args:
            document (Document) :
                A Bokeh document that is to be updated.

            callback (SessionCallback) :
                The callback to add

        """
        super().__init__(document)
        self.callback = callback

    def dispatch(self, receiver: Any) -> None:
        """ Dispatch handling of this event to a receiver.

        This method will invoke ``receiver._session_callback_added`` if
        it exists.

        """
        super().dispatch(receiver)
        if hasattr(receiver, '_session_callback_added'):
            cast(SessionCallbackAddedMixin, receiver)._session_callback_added(self)