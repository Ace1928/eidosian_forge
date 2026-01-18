from __future__ import annotations
import logging
from typing import (
import param
from ..io.resources import CDN_DIST
from ..io.state import state
from ..layout import Card, HSpacer, Row
from ..reactive import ReactiveHTML
from .terminal import Terminal
@param.depends('level', watch=True)
def _update_level(self):
    self.stream_handler.setLevel(self.level)