from __future__ import annotations
from collections.abc import Callable, Mapping
from io import SEEK_SET, UnsupportedOperation
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, cast
from .. import (
from ..abc import ByteReceiveStream, ByteSendStream
class FileWriteStream(_BaseFileStream, ByteSendStream):
    """
    A byte stream that writes to a file in the file system.

    :param file: a file that has been opened for writing in binary mode

    .. versionadded:: 3.0
    """

    @classmethod
    async def from_path(cls, path: str | PathLike[str], append: bool=False) -> FileWriteStream:
        """
        Create a file write stream by opening the given file for writing.

        :param path: path of the file to write to
        :param append: if ``True``, open the file for appending; if ``False``, any
            existing file at the given path will be truncated

        """
        mode = 'ab' if append else 'wb'
        file = await to_thread.run_sync(Path(path).open, mode)
        return cls(cast(BinaryIO, file))

    async def send(self, item: bytes) -> None:
        try:
            await to_thread.run_sync(self._file.write, item)
        except ValueError:
            raise ClosedResourceError from None
        except OSError as exc:
            raise BrokenResourceError from exc