from __future__ import annotations
from typing import Any, Optional, cast
from typing_extensions import Literal
import httpx
from ._utils import is_dict
class APIConnectionError(APIError):

    def __init__(self, *, message: str='Connection error.', request: httpx.Request) -> None:
        super().__init__(message, request, body=None)