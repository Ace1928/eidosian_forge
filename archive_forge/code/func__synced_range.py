from __future__ import annotations
from typing import (
import param
from ..models import Feed as PnFeed
from ..models.feed import ScrollButtonClick
from ..util import edit_readonly
from .base import Column
@property
def _synced_range(self):
    n = len(self.objects)
    if self.visible_range:
        return (max(self.visible_range[0] - self.load_buffer, 0), min(self.visible_range[-1] + self.load_buffer, n))
    elif self.view_latest:
        return (max(n - self.load_buffer * 2, 0), n)
    else:
        return (0, min(self.load_buffer, n))