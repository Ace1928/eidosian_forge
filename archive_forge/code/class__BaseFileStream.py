from __future__ import annotations
from collections.abc import Callable, Mapping
from io import SEEK_SET, UnsupportedOperation
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, cast
from .. import (
from ..abc import ByteReceiveStream, ByteSendStream
class _BaseFileStream:

    def __init__(self, file: BinaryIO):
        self._file = file

    async def aclose(self) -> None:
        await to_thread.run_sync(self._file.close)

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        attributes: dict[Any, Callable[[], Any]] = {FileStreamAttribute.file: lambda: self._file}
        if hasattr(self._file, 'name'):
            attributes[FileStreamAttribute.path] = lambda: Path(self._file.name)
        try:
            self._file.fileno()
        except UnsupportedOperation:
            pass
        else:
            attributes[FileStreamAttribute.fileno] = lambda: self._file.fileno()
        return attributes