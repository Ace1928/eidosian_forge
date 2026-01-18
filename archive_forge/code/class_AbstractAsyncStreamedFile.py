import asyncio
import asyncio.events
import functools
import inspect
import io
import numbers
import os
import re
import threading
from contextlib import contextmanager
from glob import has_magic
from typing import TYPE_CHECKING, Iterable
from .callbacks import DEFAULT_CALLBACK
from .exceptions import FSTimeoutError
from .implementations.local import LocalFileSystem, make_path_posix, trailing_sep
from .spec import AbstractBufferedFile, AbstractFileSystem
from .utils import glob_translate, is_exception, other_paths
class AbstractAsyncStreamedFile(AbstractBufferedFile):

    async def read(self, length=-1):
        """
        Return data from cache, or fetch pieces as necessary

        Parameters
        ----------
        length: int (-1)
            Number of bytes to read; if <0, all remaining bytes.
        """
        length = -1 if length is None else int(length)
        if self.mode != 'rb':
            raise ValueError('File not in read mode')
        if length < 0:
            length = self.size - self.loc
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        if length == 0:
            return b''
        out = await self._fetch_range(self.loc, self.loc + length)
        self.loc += len(out)
        return out

    async def write(self, data):
        """
        Write data to buffer.

        Buffer only sent on flush() or if buffer is greater than
        or equal to blocksize.

        Parameters
        ----------
        data: bytes
            Set of bytes to be written.
        """
        if self.mode not in {'wb', 'ab'}:
            raise ValueError('File not in write mode')
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        if self.forced:
            raise ValueError('This file has been force-flushed, can only close')
        out = self.buffer.write(data)
        self.loc += out
        if self.buffer.tell() >= self.blocksize:
            await self.flush()
        return out

    async def close(self):
        """Close file

        Finalizes writes, discards cache
        """
        if getattr(self, '_unclosable', False):
            return
        if self.closed:
            return
        if self.mode == 'rb':
            self.cache = None
        else:
            if not self.forced:
                await self.flush(force=True)
            if self.fs is not None:
                self.fs.invalidate_cache(self.path)
                self.fs.invalidate_cache(self.fs._parent(self.path))
        self.closed = True

    async def flush(self, force=False):
        if self.closed:
            raise ValueError('Flush on closed file')
        if force and self.forced:
            raise ValueError('Force flush cannot be called more than once')
        if force:
            self.forced = True
        if self.mode not in {'wb', 'ab'}:
            return
        if not force and self.buffer.tell() < self.blocksize:
            return
        if self.offset is None:
            self.offset = 0
            try:
                await self._initiate_upload()
            except:
                self.closed = True
                raise
        if await self._upload_chunk(final=force) is not False:
            self.offset += self.buffer.seek(0, 2)
            self.buffer = io.BytesIO()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _fetch_range(self, start, end):
        raise NotImplementedError

    async def _initiate_upload(self):
        pass

    async def _upload_chunk(self, final=False):
        raise NotImplementedError