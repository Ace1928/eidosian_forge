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
def add_get(self, path: str, handler: Handler, *, name: Optional[str]=None, allow_head: bool=True, **kwargs: Any) -> AbstractRoute:
    """Shortcut for add_route with method GET.

        If allow_head is true, another
        route is added allowing head requests to the same endpoint.
        """
    resource = self.add_resource(path, name=name)
    if allow_head:
        resource.add_route(hdrs.METH_HEAD, handler, **kwargs)
    return resource.add_route(hdrs.METH_GET, handler, **kwargs)