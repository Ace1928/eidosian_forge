from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, Optional, Pattern
import re
import time
from pyglet import clock
from pyglet import event
from pyglet.window import key
def _blink(self, dt: float) -> None:
    if self.PERIOD:
        self._blink_visible = not self._blink_visible
    if self._visible and self._active and self._blink_visible:
        alpha = self._visible_alpha
    else:
        alpha = 0
    self._list.colors[3] = alpha
    self._list.colors[7] = alpha