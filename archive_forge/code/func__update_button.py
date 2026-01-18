from __future__ import annotations
import sys
import traceback as tb
from collections import defaultdict
from typing import ClassVar, Tuple
import param
from .layout import Column, Row
from .pane import HoloViews, Markdown
from .param import Param
from .util import param_reprs
from .viewable import Viewer
from .widgets import Button, Select
def _update_button(self):
    stage, kwargs = self._stages[self._stage]
    options = list(self._graph.get(self._stage, []))
    next_param = kwargs.get('next_parameter', self.next_parameter)
    option = getattr(self._state, next_param) if next_param and next_param in stage.param else None
    if option is None:
        option = options[0] if options else None
    self.next_selector.options = options
    self.next_selector.value = option
    self.next_selector.disabled = not bool(options)
    previous = []
    for src, tgts in self._graph.items():
        if self._stage in tgts:
            previous.append(src)
    self.prev_selector.options = previous
    self.prev_selector.value = self._route[-1] if previous else None
    self.prev_selector.disabled = not bool(previous)
    if self._prev_stage is None:
        self.prev_button.disabled = True
    else:
        self.prev_button.disabled = False
    if self._next_stage is None:
        self.next_button.disabled = True
    else:
        ready = kwargs.get('ready_parameter', self.ready_parameter)
        disabled = not getattr(stage, ready) if ready in stage.param else False
        self.next_button.disabled = disabled