import asyncio
from functools import partial, singledispatch
from io import BufferedRandom, BufferedReader, BufferedWriter, FileIO, TextIOBase
from tempfile import NamedTemporaryFile as syncNamedTemporaryFile
from tempfile import SpooledTemporaryFile as syncSpooledTemporaryFile
from tempfile import TemporaryDirectory as syncTemporaryDirectory
from tempfile import TemporaryFile as syncTemporaryFile
from tempfile import _TemporaryFileWrapper as syncTemporaryFileWrapper
from ..base import AiofilesContextManager
from ..threadpool.binary import AsyncBufferedIOBase, AsyncBufferedReader, AsyncFileIO
from ..threadpool.text import AsyncTextIOWrapper
from .temptypes import AsyncSpooledTemporaryFile, AsyncTemporaryDirectory
import sys
class AiofilesContextManagerTempDir(AiofilesContextManager):
    """With returns the directory location, not the object (matching sync lib)"""

    async def __aenter__(self):
        self._obj = await self._coro
        return self._obj.name