from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar
import attrs
import trio
from trio._util import final
from .abc import AsyncResource, HalfCloseableStream, ReceiveStream, SendStream
def _is_halfclosable(stream: SendStream) -> TypeGuard[HalfCloseableStream]:
    """Check if the stream has a send_eof() method."""
    return hasattr(stream, 'send_eof')