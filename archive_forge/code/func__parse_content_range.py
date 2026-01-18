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
def _parse_content_range(self, headers):
    """Parse the Content-Range header"""
    s = headers.get('Content-Range', '')
    m = re.match('bytes (\\d+-\\d+|\\*)/(\\d+|\\*)', s)
    if not m:
        return (None, None, None)
    if m[1] == '*':
        start = end = None
    else:
        start, end = [int(x) for x in m[1].split('-')]
    total = None if m[2] == '*' else int(m[2])
    return (start, end, total)