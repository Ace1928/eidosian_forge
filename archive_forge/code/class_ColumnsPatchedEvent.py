from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class ColumnsPatchedEvent(DocumentPatchedEvent):
    """ A concrete event representing efficiently applying data patches
    to a :class:`~bokeh.models.sources.ColumnDataSource`

    """
    kind = 'ColumnsPatched'

    def __init__(self, document: Document, model: Model, attr: str, patches: Patches, setter: Setter | None=None, callback_invoker: Invoker | None=None):
        """

        Args:
            document (Document) :
                A Bokeh document that is to be updated.

            column_source (ColumnDataSource) :
                The data source to apply patches to.

            patches (list) :

            setter (ClientSession or ServerSession or None, optional) :
                This is used to prevent "boomerang" updates to Bokeh apps.
                (default: None)

                See :class:`~bokeh.document.events.DocumentChangedEvent`
                for more details.

            callback_invoker (callable, optional) :
                A callable that will invoke any Model callbacks that should
                be executed in response to the change that triggered this
                event. (default: None)

        """
        super().__init__(document, setter, callback_invoker)
        self.model = model
        self.attr = attr
        self.patches = patches

    def dispatch(self, receiver: Any) -> None:
        """ Dispatch handling of this event to a receiver.

        This method will invoke ``receiver._columns_patched`` if it exists.

        """
        super().dispatch(receiver)
        if hasattr(receiver, '_columns_patched'):
            cast(ColumnsPatchedMixin, receiver)._columns_patched(self)

    def to_serializable(self, serializer: Serializer) -> ColumnsPatched:
        """ Create a JSON representation of this event suitable for sending
        to clients.

        .. code-block:: python

            {
                'kind'          : 'ColumnsPatched'
                'column_source' : <reference to a CDS>
                'patches'       : <patches to apply to column_source>
            }

        Args:
            serializer (Serializer):

        """
        return ColumnsPatched(kind=self.kind, model=self.model.ref, attr=self.attr, patches=serializer.encode(self.patches))

    @staticmethod
    def _handle_event(doc: Document, event: ColumnsPatchedEvent) -> None:
        model = event.model
        attr = event.attr
        assert attr == 'data'
        patches = event.patches
        model.patch(patches, event.setter)