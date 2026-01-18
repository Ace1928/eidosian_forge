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
class ResourcesView(Sized, Iterable[AbstractResource], Container[AbstractResource]):

    def __init__(self, resources: List[AbstractResource]) -> None:
        self._resources = resources

    def __len__(self) -> int:
        return len(self._resources)

    def __iter__(self) -> Iterator[AbstractResource]:
        yield from self._resources

    def __contains__(self, resource: object) -> bool:
        return resource in self._resources