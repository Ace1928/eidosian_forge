from typing import Any, Iterator, Optional, Sequence
from ..utils.base64 import base64, unbase64
from .connection import (
Get offset from a given cursor and a default.

    Given an optional cursor and a default offset, return the offset to use;
    if the cursor contains a valid offset, that will be used,
    otherwise it will be the default.
    