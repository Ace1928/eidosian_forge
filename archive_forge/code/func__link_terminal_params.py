from __future__ import annotations
import sys
from typing import TYPE_CHECKING, ClassVar
import param
from ..io.state import state
from ..viewable import Viewable
from ..widgets import Terminal
from .base import PaneBase
def _link_terminal_params(self, *events):
    self._terminal.param.update({event.name: event.new for event in events})