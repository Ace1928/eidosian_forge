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
def _replace_placeholder(self, message: ChatMessage | None=None) -> None:
    """
        Replace the placeholder from the chat log with the message
        if placeholder, otherwise simply append the message.
        Replacing helps lessen the chat log jumping around.
        """
    index = None
    if self.placeholder_threshold > 0:
        try:
            index = self.index(self._placeholder)
        except ValueError:
            pass
    if index is not None:
        if message is not None:
            self[index] = message
        elif message is None:
            self.remove(self._placeholder)
    elif message is not None:
        self.append(message)