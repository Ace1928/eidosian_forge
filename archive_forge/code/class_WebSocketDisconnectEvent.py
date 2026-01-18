from __future__ import annotations
import sys
import types
from typing import (
class WebSocketDisconnectEvent(TypedDict):
    type: Literal['websocket.disconnect']
    code: int