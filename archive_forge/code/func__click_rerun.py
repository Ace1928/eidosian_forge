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
def _click_rerun(self, event: param.parameterized.Event | None=None, instance: 'ChatInterface' | None=None) -> None:
    """
        Upon clicking the rerun button, rerun the last user message,
        which can trigger the callback again.
        """
    count = self._get_last_user_entry_index()
    messages = self.undo(count)
    if not messages:
        return
    self.send(value=messages[0], respond=True)