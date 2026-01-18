from __future__ import annotations
import logging
from typing import (
import param
from ..io.resources import CDN_DIST
from ..io.state import state
from ..layout import Card, HSpacer, Row
from ..reactive import ReactiveHTML
from .terminal import Terminal
def _update_debugger(self, record):
    if not hasattr(self, 'debugger'):
        return
    if record.levelno >= 40:
        self.debugger._number_of_errors += 1
    elif 40 > record.levelno >= 30:
        self.debugger._number_of_warnings += 1
    elif record.levelno < 30:
        self.debugger._number_of_infos += 1