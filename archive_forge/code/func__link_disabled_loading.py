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
def _link_disabled_loading(self, obj: Viewable):
    """
        Link the disabled and loading attributes of the chat box to the
        given object.
        """
    for attr in ['disabled', 'loading']:
        setattr(obj, attr, getattr(self, attr))
        self.link(obj, **{attr: attr})