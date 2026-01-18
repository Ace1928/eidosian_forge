from __future__ import annotations
import logging # isort:skip
from typing import (
class MessageSent(TypedDict):
    kind: Literal['MessageSent']
    msg_type: str
    msg_data: Any | None