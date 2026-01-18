from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from ..abc import (
@dataclass(eq=False)
class StapledByteStream(ByteStream):
    """
    Combines two byte streams into a single, bidirectional byte stream.

    Extra attributes will be provided from both streams, with the receive stream
    providing the values in case of a conflict.

    :param ByteSendStream send_stream: the sending byte stream
    :param ByteReceiveStream receive_stream: the receiving byte stream
    """
    send_stream: ByteSendStream
    receive_stream: ByteReceiveStream

    async def receive(self, max_bytes: int=65536) -> bytes:
        return await self.receive_stream.receive(max_bytes)

    async def send(self, item: bytes) -> None:
        await self.send_stream.send(item)

    async def send_eof(self) -> None:
        await self.send_stream.aclose()

    async def aclose(self) -> None:
        await self.send_stream.aclose()
        await self.receive_stream.aclose()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {**self.send_stream.extra_attributes, **self.receive_stream.extra_attributes}