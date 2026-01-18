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
def _click_undo(self, event: param.parameterized.Event | None=None, instance: 'ChatInterface' | None=None) -> None:
    """
        Upon clicking the undo button, undo (remove) messages
        up to the last user message. If the button is clicked
        again without performing any other actions, revert the undo.
        """
    undo_data = self._button_data['undo']
    undo_objects = undo_data.objects
    if not undo_objects:
        self._reset_button_data()
        count = self._get_last_user_entry_index()
        undo_data.objects = self.undo(count)
        if self._allow_revert:
            self._toggle_revert(undo_data, True)
        else:
            undo_data.objects = []
    else:
        self.extend(undo_objects)
        self._reset_button_data()