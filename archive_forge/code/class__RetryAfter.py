from __future__ import annotations
import typing as t
from datetime import datetime
from markupsafe import escape
from markupsafe import Markup
from ._internal import _get_environ
class _RetryAfter(HTTPException):
    """Adds an optional ``retry_after`` parameter which will set the
    ``Retry-After`` header. May be an :class:`int` number of seconds or
    a :class:`~datetime.datetime`.
    """

    def __init__(self, description: str | None=None, response: Response | None=None, retry_after: datetime | int | None=None) -> None:
        super().__init__(description, response)
        self.retry_after = retry_after

    def get_headers(self, environ: WSGIEnvironment | None=None, scope: dict | None=None) -> list[tuple[str, str]]:
        headers = super().get_headers(environ, scope)
        if self.retry_after:
            if isinstance(self.retry_after, datetime):
                from .http import http_date
                value = http_date(self.retry_after)
            else:
                value = str(self.retry_after)
            headers.append(('Retry-After', value))
        return headers