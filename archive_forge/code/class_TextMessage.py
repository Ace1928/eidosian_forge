from abc import ABC
from dataclasses import dataclass, field
from typing import Generic, List, Optional, Sequence, TypeVar, Union
from .extensions import Extension
from .typing import Headers
@dataclass(frozen=True)
class TextMessage(Message[str]):
    """This event is fired when a data frame with TEXT payload is received.

    Fields:

    .. attribute:: data

       The message data as string, This only represents a single chunk
       of data and not a full WebSocket message.  You need to buffer
       and reassemble these chunks to get the full message.

    """
    data: str