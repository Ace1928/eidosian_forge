from __future__ import annotations
import asyncio
import traceback
from enum import Enum
from inspect import (
from io import BytesIO
from typing import (
import param
from .._param import Margin
from ..io.resources import CDN_DIST
from ..layout import Feed, ListPanel
from ..layout.card import Card
from ..layout.spacer import VSpacer
from ..pane.image import SVG
from .message import ChatMessage
def _gather_callback_args(self, message: ChatMessage) -> Any:
    """
        Extracts the contents from the message's panel object.
        """
    value = message._object_panel
    if hasattr(value, 'object'):
        contents = value.object
    elif hasattr(value, 'objects'):
        contents = value.objects
    elif hasattr(value, 'value'):
        contents = value.value
    else:
        contents = value
    return (contents, message.user, self)