import abc
import asyncio
import base64
import hashlib
import inspect
import keyword
import os
import re
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import (
from yarl import URL, __version__ as yarl_version  # type: ignore[attr-defined]
from . import hdrs
from .abc import AbstractMatchInfo, AbstractRouter, AbstractView
from .helpers import DEBUG
from .http import HttpVersion11
from .typedefs import Handler, PathLike
from .web_exceptions import (
from .web_fileresponse import FileResponse
from .web_request import Request
from .web_response import Response, StreamResponse
from .web_routedef import AbstractRouteDef
class SystemRoute(AbstractRoute):

    def __init__(self, http_exception: HTTPException) -> None:
        super().__init__(hdrs.METH_ANY, self._handle)
        self._http_exception = http_exception

    def url_for(self, *args: str, **kwargs: str) -> URL:
        raise RuntimeError('.url_for() is not allowed for SystemRoute')

    @property
    def name(self) -> Optional[str]:
        return None

    def get_info(self) -> _InfoDict:
        return {'http_exception': self._http_exception}

    async def _handle(self, request: Request) -> StreamResponse:
        raise self._http_exception

    @property
    def status(self) -> int:
        return self._http_exception.status

    @property
    def reason(self) -> str:
        return self._http_exception.reason

    def __repr__(self) -> str:
        return '<SystemRoute {self.status}: {self.reason}>'.format(self=self)