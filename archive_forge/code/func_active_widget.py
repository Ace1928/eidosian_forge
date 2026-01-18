from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import (
import param
from ..io.resources import CDN_DIST
from ..layout import Row, Tabs
from ..pane.image import ImageBase
from ..viewable import Viewable
from ..widgets.base import Widget
from ..widgets.button import Button
from ..widgets.input import FileInput, TextInput
from .feed import CallbackState, ChatFeed
from .input import ChatAreaInput
from .message import ChatMessage, _FileInputMessage
@property
def active_widget(self) -> Widget:
    """
        The currently active widget.

        Returns
        -------
        The active widget.
        """
    if isinstance(self._input_layout, Tabs):
        return self._input_layout[self.active].objects[0]
    return self._input_layout.objects[0]