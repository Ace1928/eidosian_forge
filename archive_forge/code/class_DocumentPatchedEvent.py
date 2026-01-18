from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class DocumentPatchedEvent(DocumentChangedEvent, Serializable):
    """ A Base class for events that represent updating Bokeh Models and
    their properties.

    """
    kind: ClassVar[str]
    _handlers: ClassVar[dict[str, type[DocumentPatchedEvent]]] = {}

    def __init_subclass__(cls):
        cls._handlers[cls.kind] = cls

    def dispatch(self, receiver: Any) -> None:
        """ Dispatch handling of this event to a receiver.

        This method will invoke ``receiver._document_patched`` if it exists.

        """
        super().dispatch(receiver)
        if hasattr(receiver, '_document_patched'):
            cast(DocumentPatchedMixin, receiver)._document_patched(self)

    def to_serializable(self, serializer: Serializer) -> DocumentPatched:
        """ Create a JSON representation of this event suitable for sending
        to clients.

        *Sub-classes must implement this method.*

        Args:
            serializer (Serializer):

        """
        raise NotImplementedError()

    @staticmethod
    def handle_event(doc: Document, event_rep: DocumentPatched, setter: Setter | None) -> None:
        """

        """
        event_kind = event_rep.pop('kind')
        event_cls = DocumentPatchedEvent._handlers.get(event_kind, None)
        if event_cls is None:
            raise RuntimeError(f"unknown patch event type '{event_kind!r}'")
        event = event_cls(document=doc, setter=setter, **event_rep)
        event_cls._handle_event(doc, event)

    @staticmethod
    def _handle_event(doc: Document, event: DocumentPatchedEvent) -> None:
        raise NotImplementedError()