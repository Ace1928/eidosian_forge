import asyncio
import zlib
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Union  # noqa
from multidict import CIMultiDict
from .abc import AbstractStreamWriter
from .base_protocol import BaseProtocol
from .compression_utils import ZLibCompressor
from .helpers import NO_EXTENSIONS
class HttpVersion(NamedTuple):
    major: int
    minor: int