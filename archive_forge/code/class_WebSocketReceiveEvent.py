import sys
from typing import (
class WebSocketReceiveEvent(TypedDict):
    type: Literal['websocket.receive']
    bytes: Optional[bytes]
    text: Optional[str]