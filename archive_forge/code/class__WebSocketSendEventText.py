from __future__ import annotations
import sys
import types
from typing import (
class _WebSocketSendEventText(TypedDict):
    type: Literal['websocket.send']
    bytes: NotRequired[None]
    text: str