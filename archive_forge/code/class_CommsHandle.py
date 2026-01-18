from __future__ import annotations
import logging # isort:skip
import json
import os
import urllib
from typing import (
from uuid import uuid4
from ..core.types import ID
from ..util.serialization import make_id
from ..util.warnings import warn
from .state import curstate
class CommsHandle:
    """

    """
    _json: Any = {}
    _cellno: int | None
    _doc: Document

    def __init__(self, comms: Comm, cell_doc: Document) -> None:
        self._cellno = None
        try:
            from IPython import get_ipython
            ip = get_ipython()
            assert ip is not None
            hm = ip.history_manager
            assert hm is not None
            p_prompt = next(iter(hm.get_tail(1, include_latest=True)))[1]
            self._cellno = p_prompt
        except Exception as e:
            log.debug('Could not get Notebook cell number, reason: %s', e)
        self._comms = comms
        self._doc = cell_doc
        self._doc.hold()

    def _repr_html_(self) -> str:
        if self._cellno is not None:
            return f'<p><code>&lt;Bokeh Notebook handle for <strong>In[{self._cellno}]</strong>&gt;</code></p>'
        else:
            return '<p><code>&lt;Bokeh Notebook handle&gt;</code></p>'

    @property
    def comms(self) -> Comm:
        return self._comms

    @property
    def doc(self) -> Document:
        return self._doc

    def _document_model_changed(self, event: ModelChangedEvent) -> None:
        if event.model.id in self.doc.models:
            self.doc.callbacks.trigger_on_change(event)

    def _column_data_changed(self, event: ColumnDataChangedEvent) -> None:
        if event.model.id in self.doc.models:
            self.doc.callbacks.trigger_on_change(event)

    def _columns_streamed(self, event: ColumnsStreamedEvent) -> None:
        if event.model.id in self.doc.models:
            self.doc.callbacks.trigger_on_change(event)

    def _columns_patched(self, event: ColumnsPatchedEvent) -> None:
        if event.model.id in self.doc.models:
            self.doc.callbacks.trigger_on_change(event)