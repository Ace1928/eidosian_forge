from __future__ import annotations
import typing
import warnings
from os import PathLike
from starlette.background import BackgroundTask
from starlette.datastructures import URL
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.types import Receive, Scope, Send
def _create_env(self, directory: str | PathLike[typing.AnyStr] | typing.Sequence[str | PathLike[typing.AnyStr]], **env_options: typing.Any) -> jinja2.Environment:
    loader = jinja2.FileSystemLoader(directory)
    env_options.setdefault('loader', loader)
    env_options.setdefault('autoescape', True)
    return jinja2.Environment(**env_options)