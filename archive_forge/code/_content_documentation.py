from __future__ import annotations
import inspect
import warnings
from json import dumps as json_dumps
from typing import (
from urllib.parse import urlencode
from ._exceptions import StreamClosed, StreamConsumed
from ._multipart import MultipartStream
from ._types import (
from ._utils import peek_filelike_length, primitive_value_to_str

    Handles encoding the given `content`, returning a two-tuple of
    (<headers>, <stream>).
    