from __future__ import annotations
import types
from collections.abc import Iterable, Mapping
from inspect import (
from typing import TYPE_CHECKING, ClassVar
import param
from .layout import Column, Panel, Row
from .pane import HTML, PaneBase, panel
from .pane.base import ReplacementPane
from .viewable import Viewable
from .widgets import Button, Widget
from .widgets.widget import fixed, widget
def _link_widgets(self):
    if self.manual_update:
        widgets = [('manual', self._widgets['manual'])]
    else:
        widgets = self._widgets.items()
    for name, widget_obj in widgets:

        def update_pane(change):
            new_object = self.object(**self.kwargs)
            new_pane, internal = ReplacementPane._update_from_object(new_object, self._pane, self._internal)
            if new_pane is None:
                return
            self._pane = new_pane
            self._inner_layout[0] = new_pane
            self._internal = internal
        if self.throttled and hasattr(widget_obj, 'value_throttled'):
            v = 'value_throttled'
        else:
            v = 'value'
        pname = 'clicks' if name == 'manual' else v
        watcher = widget_obj.param.watch(update_pane, pname)
        self._internal_callbacks.append(watcher)