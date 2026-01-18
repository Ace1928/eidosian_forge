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
class ResourceRoute(AbstractRoute):
    """A route with resource"""

    def __init__(self, method: str, handler: Union[Handler, Type[AbstractView]], resource: AbstractResource, *, expect_handler: Optional[_ExpectHandler]=None) -> None:
        super().__init__(method, handler, expect_handler=expect_handler, resource=resource)

    def __repr__(self) -> str:
        return '<ResourceRoute [{method}] {resource} -> {handler!r}'.format(method=self.method, resource=self._resource, handler=self.handler)

    @property
    def name(self) -> Optional[str]:
        if self._resource is None:
            return None
        return self._resource.name

    def url_for(self, *args: str, **kwargs: str) -> URL:
        """Construct url for route with additional params."""
        assert self._resource is not None
        return self._resource.url_for(*args, **kwargs)

    def get_info(self) -> _InfoDict:
        assert self._resource is not None
        return self._resource.get_info()