from __future__ import annotations
import contextlib
import functools
import inspect
import re
import traceback
import types
import typing
import warnings
from contextlib import asynccontextmanager
from enum import Enum
from starlette._exception_handler import wrap_app_handling_exceptions
from starlette._utils import get_route_path, is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.convertors import CONVERTOR_TYPES, Convertor
from starlette.datastructures import URL, Headers, URLPath
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketClose
class Route(BaseRoute):

    def __init__(self, path: str, endpoint: typing.Callable[..., typing.Any], *, methods: list[str] | None=None, name: str | None=None, include_in_schema: bool=True, middleware: typing.Sequence[Middleware] | None=None) -> None:
        assert path.startswith('/'), "Routed paths must start with '/'"
        self.path = path
        self.endpoint = endpoint
        self.name = get_name(endpoint) if name is None else name
        self.include_in_schema = include_in_schema
        endpoint_handler = endpoint
        while isinstance(endpoint_handler, functools.partial):
            endpoint_handler = endpoint_handler.func
        if inspect.isfunction(endpoint_handler) or inspect.ismethod(endpoint_handler):
            self.app = request_response(endpoint)
            if methods is None:
                methods = ['GET']
        else:
            self.app = endpoint
        if middleware is not None:
            for cls, args, kwargs in reversed(middleware):
                self.app = cls(*args, app=self.app, **kwargs)
        if methods is None:
            self.methods = None
        else:
            self.methods = {method.upper() for method in methods}
            if 'GET' in self.methods:
                self.methods.add('HEAD')
        self.path_regex, self.path_format, self.param_convertors = compile_path(path)

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        path_params: 'typing.Dict[str, typing.Any]'
        if scope['type'] == 'http':
            route_path = get_route_path(scope)
            match = self.path_regex.match(route_path)
            if match:
                matched_params = match.groupdict()
                for key, value in matched_params.items():
                    matched_params[key] = self.param_convertors[key].convert(value)
                path_params = dict(scope.get('path_params', {}))
                path_params.update(matched_params)
                child_scope = {'endpoint': self.endpoint, 'path_params': path_params}
                if self.methods and scope['method'] not in self.methods:
                    return (Match.PARTIAL, child_scope)
                else:
                    return (Match.FULL, child_scope)
        return (Match.NONE, {})

    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath:
        seen_params = set(path_params.keys())
        expected_params = set(self.param_convertors.keys())
        if name != self.name or seen_params != expected_params:
            raise NoMatchFound(name, path_params)
        path, remaining_params = replace_params(self.path_format, self.param_convertors, path_params)
        assert not remaining_params
        return URLPath(path=path, protocol='http')

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods and scope['method'] not in self.methods:
            headers = {'Allow': ', '.join(self.methods)}
            if 'app' in scope:
                raise HTTPException(status_code=405, headers=headers)
            else:
                response = PlainTextResponse('Method Not Allowed', status_code=405, headers=headers)
            await response(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    def __eq__(self, other: typing.Any) -> bool:
        return isinstance(other, Route) and self.path == other.path and (self.endpoint == other.endpoint) and (self.methods == other.methods)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        methods = sorted(self.methods or [])
        path, name = (self.path, self.name)
        return f'{class_name}(path={path!r}, name={name!r}, methods={methods!r})'