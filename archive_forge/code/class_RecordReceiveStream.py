from __future__ import annotations
from typing import NoReturn
import attrs
import pytest
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
@attrs.define(slots=False)
class RecordReceiveStream(ReceiveStream):
    record: list[str | tuple[str, int | None]] = attrs.Factory(list)

    async def receive_some(self, max_bytes: int | None=None) -> bytes:
        self.record.append(('receive_some', max_bytes))
        return b''

    async def aclose(self) -> None:
        self.record.append('aclose')