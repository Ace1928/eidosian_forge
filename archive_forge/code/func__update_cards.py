from __future__ import annotations
from typing import (
import param
from bokeh.models import Column as BkColumn, CustomJS
from ..reactive import Reactive
from .base import NamedListPanel
from .card import Card
def _update_cards(self, *events):
    params = {k: v for k, v in self.param.values().items() if k in self._synced_properties}
    for panel in self._panels.values():
        panel.param.update(**params)