from __future__ import annotations
from typing import NoReturn
import attrs
import pytest
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
class BrokenReceiveStream(RecordReceiveStream):

    async def aclose(self) -> NoReturn:
        await super().aclose()
        raise ValueError('recv error')