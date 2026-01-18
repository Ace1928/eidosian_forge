import asyncio
import io
import logging
import re
import weakref
from copy import copy
from urllib.parse import urlparse
import aiohttp
import yarl
from fsspec.asyn import AbstractAsyncStreamedFile, AsyncFileSystem, sync, sync_wrapper
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.exceptions import FSTimeoutError
from fsspec.spec import AbstractBufferedFile
from fsspec.utils import (
from ..caching import AllBytes
class AsyncStreamFile(AbstractAsyncStreamedFile):

    def __init__(self, fs, url, mode='rb', loop=None, session=None, size=None, **kwargs):
        self.url = url
        self.session = session
        self.r = None
        if mode != 'rb':
            raise ValueError
        self.details = {'name': url, 'size': None}
        self.kwargs = kwargs
        super().__init__(fs=fs, path=url, mode=mode, cache_type='none')
        self.size = size

    async def read(self, num=-1):
        if self.r is None:
            r = await self.session.get(self.fs.encode_url(self.url), **self.kwargs).__aenter__()
            self.fs._raise_not_found_for_status(r, self.url)
            self.r = r
        out = await self.r.content.read(num)
        self.loc += len(out)
        return out

    async def close(self):
        if self.r is not None:
            self.r.close()
            self.r = None
        await super().close()