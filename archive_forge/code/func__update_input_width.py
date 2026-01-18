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
@param.depends('width', watch=True)
def _update_input_width(self):
    """
        Update the input width.
        """
    if self.show_button_name is None:
        self.show_button_name = self.width is None or self.width >= 400