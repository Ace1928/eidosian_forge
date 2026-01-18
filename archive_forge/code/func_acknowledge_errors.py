from __future__ import annotations
import logging
from typing import (
import param
from ..io.resources import CDN_DIST
from ..io.state import state
from ..layout import Card, HSpacer, Row
from ..reactive import ReactiveHTML
from .terminal import Terminal
def acknowledge_errors(self, event):
    self._number_of_errors = 0
    self._number_of_warnings = 0
    self._number_of_infos = 0