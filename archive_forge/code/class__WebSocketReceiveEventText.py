from __future__ import annotations
import sys
import types
from typing import (
class _WebSocketReceiveEventText(TypedDict):
    type: Literal['websocket.receive']
    bytes: NotRequired[None]
    text: str