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
class MatchedSubAppResource(PrefixedSubAppResource):

    def __init__(self, rule: AbstractRuleMatching, app: 'Application') -> None:
        AbstractResource.__init__(self)
        self._prefix = ''
        self._app = app
        self._rule = rule

    @property
    def canonical(self) -> str:
        return self._rule.canonical

    def get_info(self) -> _InfoDict:
        return {'app': self._app, 'rule': self._rule}

    async def resolve(self, request: Request) -> _Resolve:
        if not await self._rule.match(request):
            return (None, set())
        match_info = await self._app.router.resolve(request)
        match_info.add_app(self._app)
        if isinstance(match_info.http_exception, HTTPMethodNotAllowed):
            methods = match_info.http_exception.allowed_methods
        else:
            methods = set()
        return (match_info, methods)

    def __repr__(self) -> str:
        return '<MatchedSubAppResource -> {app!r}>'.format(app=self._app)