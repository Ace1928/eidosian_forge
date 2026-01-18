from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class DocumentReady(DocumentEvent):
    """
    Announce when a Document is fully idle.

    .. note::
        To register a JS callback for this event in standalone embedding
        mode, one has to either use ``curdoc()`` or an explicit ``Document``
        instance, e.g.:

        .. code-block:: python

            from bokeh.io import curdoc
            curdoc().js_on_event("document_ready", handler)

    """
    event_name = 'document_ready'