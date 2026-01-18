import asyncio
import dataclasses
import email.message
import inspect
import json
from contextlib import AsyncExitStack
from enum import Enum, IntEnum
from typing import (
from fastapi import params
from fastapi._compat import (
from fastapi.datastructures import Default, DefaultPlaceholder
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import (
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import (
from fastapi.types import DecoratedCallable, IncEx
from fastapi.utils import (
from pydantic import BaseModel
from starlette import routing
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import (
from starlette.routing import Mount as Mount  # noqa
from starlette.types import ASGIApp, Lifespan, Scope
from starlette.websockets import WebSocket
from typing_extensions import Annotated, Doc, deprecated  # type: ignore [attr-defined]
def add_api_websocket_route(self, path: str, endpoint: Callable[..., Any], name: Optional[str]=None, *, dependencies: Optional[Sequence[params.Depends]]=None) -> None:
    current_dependencies = self.dependencies.copy()
    if dependencies:
        current_dependencies.extend(dependencies)
    route = APIWebSocketRoute(self.prefix + path, endpoint=endpoint, name=name, dependencies=current_dependencies, dependency_overrides_provider=self.dependency_overrides_provider)
    self.routes.append(route)