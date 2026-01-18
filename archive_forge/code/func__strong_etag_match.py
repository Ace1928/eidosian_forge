import asyncio
import mimetypes
import os
import pathlib
from typing import (  # noqa
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import ETAG_ANY, ETag, must_be_empty_body
from .typedefs import LooseHeaders, PathLike
from .web_exceptions import (
from .web_response import StreamResponse
@staticmethod
def _strong_etag_match(etag_value: str, etags: Tuple[ETag, ...]) -> bool:
    if len(etags) == 1 and etags[0].value == ETAG_ANY:
        return True
    return any((etag.value == etag_value for etag in etags if not etag.is_weak))