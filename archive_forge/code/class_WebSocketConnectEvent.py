from __future__ import annotations
import sys
import types
from typing import (
class WebSocketConnectEvent(TypedDict):
    type: Literal['websocket.connect']