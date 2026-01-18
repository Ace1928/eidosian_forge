from __future__ import annotations
import typing as t
from datetime import datetime
from markupsafe import escape
from markupsafe import Markup
from ._internal import _get_environ
class RequestHeaderFieldsTooLarge(HTTPException):
    """*431* `Request Header Fields Too Large`

    The server refuses to process the request because the header fields are too
    large. One or more individual fields may be too large, or the set of all
    headers is too large.
    """
    code = 431
    description = 'One or more header fields exceeds the maximum size.'