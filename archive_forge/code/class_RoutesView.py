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
class RoutesView(Sized, Iterable[AbstractRoute], Container[AbstractRoute]):

    def __init__(self, resources: List[AbstractResource]):
        self._routes: List[AbstractRoute] = []
        for resource in resources:
            for route in resource:
                self._routes.append(route)

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self) -> Iterator[AbstractRoute]:
        yield from self._routes

    def __contains__(self, route: object) -> bool:
        return route in self._routes