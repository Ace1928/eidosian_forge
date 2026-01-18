from __future__ import annotations
from typing import (
import param
from ..models import Feed as PnFeed
from ..models.feed import ScrollButtonClick
from ..util import edit_readonly
from .base import Column
@param.depends('visible_range', 'load_buffer', watch=True)
def _trigger_get_objects(self):
    if self.visible_range is None:
        return
    vs, ve = self.visible_range
    ss, se = self._last_synced
    half_buffer = self.load_buffer // 2
    top_trigger = vs - ss < half_buffer
    bottom_trigger = se - ve < half_buffer
    invalid_trigger = ve - vs < self.load_buffer and ve - vs < len(self.objects)
    if top_trigger or bottom_trigger or invalid_trigger:
        self.param.trigger('objects')