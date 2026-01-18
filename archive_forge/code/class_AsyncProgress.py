import os
import sys
from typing import IO, TYPE_CHECKING, Optional
from wandb.errors import CommError
class AsyncProgress:
    """Wrapper around Progress, to make it async iterable.

    httpx, for streaming uploads, requires the data source to be an async iterable.
    If we pass in a sync iterable (like a bare `Progress` instance), httpx will
    get confused, think we're trying to make a synchronous request, and raise.
    So we need this wrapper class to be an async iterable but *not* a sync iterable.
    """

    def __init__(self, progress: Progress) -> None:
        self._progress = progress

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._progress)
        except StopIteration:
            raise StopAsyncIteration

    def __len__(self):
        return len(self._progress)

    def rewind(self) -> None:
        self._progress.rewind()