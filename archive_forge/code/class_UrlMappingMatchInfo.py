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
class UrlMappingMatchInfo(BaseDict, AbstractMatchInfo):

    def __init__(self, match_dict: Dict[str, str], route: AbstractRoute):
        super().__init__(match_dict)
        self._route = route
        self._apps: List[Application] = []
        self._current_app: Optional[Application] = None
        self._frozen = False

    @property
    def handler(self) -> Handler:
        return self._route.handler

    @property
    def route(self) -> AbstractRoute:
        return self._route

    @property
    def expect_handler(self) -> _ExpectHandler:
        return self._route.handle_expect_header

    @property
    def http_exception(self) -> Optional[HTTPException]:
        return None

    def get_info(self) -> _InfoDict:
        return self._route.get_info()

    @property
    def apps(self) -> Tuple['Application', ...]:
        return tuple(self._apps)

    def add_app(self, app: 'Application') -> None:
        if self._frozen:
            raise RuntimeError('Cannot change apps stack after .freeze() call')
        if self._current_app is None:
            self._current_app = app
        self._apps.insert(0, app)

    @property
    def current_app(self) -> 'Application':
        app = self._current_app
        assert app is not None
        return app

    @contextmanager
    def set_current_app(self, app: 'Application') -> Generator[None, None, None]:
        if DEBUG:
            if app not in self._apps:
                raise RuntimeError('Expected one of the following apps {!r}, got {!r}'.format(self._apps, app))
        prev = self._current_app
        self._current_app = app
        try:
            yield
        finally:
            self._current_app = prev

    def freeze(self) -> None:
        self._frozen = True

    def __repr__(self) -> str:
        return f'<MatchInfo {super().__repr__()}: {self._route}>'