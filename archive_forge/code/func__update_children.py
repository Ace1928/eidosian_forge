from __future__ import annotations
import datetime as dt
import uuid
from functools import partial
from types import FunctionType, MethodType
from typing import (
import numpy as np
import param
from bokeh.model import Model
from bokeh.models import ColumnDataSource, ImportedStyleSheet
from bokeh.models.widgets.tables import (
from bokeh.util.serialization import convert_datetime_array
from pyviz_comms import JupyterComm
from ..depends import transform_reference
from ..io.resources import CDN_DIST, CSS_URLS
from ..io.state import state
from ..reactive import Reactive, ReactiveData
from ..util import (
from ..util.warnings import warn
from .base import Widget
from .button import Button
from .input import TextInput
def _update_children(self, *events):
    cleanup, reuse = (set(), set())
    page_events = ('page', 'page_size', 'pagination')
    for event in events:
        if event.name == 'expanded' and len(events) == 1:
            cleanup = set(event.old) - set(event.new)
            reuse = set(event.old) & set(event.new)
        elif event.name == 'value' and self._indexes_changed(event.old, event.new) or (event.name in page_events and (not self._updating)) or (self.pagination == 'remote' and event.name == 'sorters'):
            self.expanded = []
            return
    old_panels = self._child_panels
    self._child_panels = child_panels = self._get_children({i: old_panels[i] for i in reuse})
    for ref, (m, _) in self._models.items():
        root, doc, comm = state._views[ref][1:]
        for idx in cleanup:
            old_panels[idx]._cleanup(root)
        children = self._get_model_children(child_panels, doc, root, m, comm)
        msg = {'children': children}
        self._apply_update([], msg, m, ref)