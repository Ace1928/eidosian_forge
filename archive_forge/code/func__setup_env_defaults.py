from __future__ import annotations
import typing
import warnings
from os import PathLike
from starlette.background import BackgroundTask
from starlette.datastructures import URL
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.types import Receive, Scope, Send
def _setup_env_defaults(self, env: jinja2.Environment) -> None:

    @pass_context
    def url_for(context: typing.Dict[str, typing.Any], name: str, /, **path_params: typing.Any) -> URL:
        request: Request = context['request']
        return request.url_for(name, **path_params)
    env.globals.setdefault('url_for', url_for)