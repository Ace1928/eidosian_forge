from __future__ import annotations
import attrs
import pytest
from .. import abc as tabc
from ..lowlevel import Task
@attrs.define(slots=False)
class MyAR(tabc.AsyncResource):
    record: list[str] = attrs.Factory(list)

    async def aclose(self) -> None:
        self.record.append('ac')