from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class ColumnDataChangedEvent(DocumentPatchedEvent):
    """ A concrete event representing efficiently replacing *all*
    existing data for a :class:`~bokeh.models.sources.ColumnDataSource`

    """
    kind = 'ColumnDataChanged'

    def __init__(self, document: Document, model: Model, attr: str, data: DataDict | None=None, cols: list[str] | None=None, setter: Setter | None=None, callback_invoker: Invoker | None=None):
        """

        Args:
            document (Document) :
                A Bokeh document that is to be updated.

            column_source (ColumnDataSource) :

            cols (list[str]) :
                optional explicit list of column names to update. If None, all
                columns will be updated (default: None)

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
        self.data = data
        self.cols = cols

    def dispatch(self, receiver: Any) -> None:
        """ Dispatch handling of this event to a receiver.

        This method will invoke ``receiver._column_data_changed`` if it exists.

        """
        super().dispatch(receiver)
        if hasattr(receiver, '_column_data_changed'):
            cast(ColumnDataChangedMixin, receiver)._column_data_changed(self)

    def to_serializable(self, serializer: Serializer) -> ColumnDataChanged:
        """ Create a JSON representation of this event suitable for sending
        to clients.

        .. code-block:: python

            {
                'kind'          : 'ColumnDataChanged'
                'column_source' : <reference to a CDS>
                'data'          : <new data to steam to column_source>
                'cols'          : <specific columns to update>
            }

        Args:
            serializer (Serializer):

        """
        data = self.data if self.data is not None else getattr(self.model, self.attr)
        cols = self.cols
        if cols is not None:
            data = {col: value for col in cols if (value := data.get(col)) is not None}
        return ColumnDataChanged(kind=self.kind, model=self.model.ref, attr=self.attr, data=serializer.encode(data), cols=serializer.encode(cols))

    @staticmethod
    def _handle_event(doc: Document, event: ColumnDataChangedEvent) -> None:
        model = event.model
        attr = event.attr
        data = event.data
        model.set_from_json(attr, data, setter=event.setter)