from __future__ import annotations
from typing import Any, Optional, cast
from typing_extensions import Literal
import httpx
from ._utils import is_dict
class APIResponseValidationError(APIError):
    response: httpx.Response
    status_code: int

    def __init__(self, response: httpx.Response, body: object | None, *, message: str | None=None) -> None:
        super().__init__(message or 'Data returned by API invalid for expected schema.', response.request, body=body)
        self.response = response
        self.status_code = response.status_code