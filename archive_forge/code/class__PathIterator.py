from __future__ import annotations
import os
import pathlib
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from os import PathLike
from typing import (
from .. import to_thread
from ..abc import AsyncResource
@dataclass(eq=False)
class _PathIterator(AsyncIterator['Path']):
    iterator: Iterator[PathLike[str]]

    async def __anext__(self) -> Path:
        nextval = await to_thread.run_sync(next, self.iterator, None, abandon_on_cancel=True)
        if nextval is None:
            raise StopAsyncIteration from None
        return Path(nextval)