from __future__ import annotations
import enum
import json
import typing
from starlette.requests import HTTPConnection
from starlette.types import Message, Receive, Scope, Send
class WebSocketState(enum.Enum):
    CONNECTING = 0
    CONNECTED = 1
    DISCONNECTED = 2