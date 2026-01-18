from __future__ import annotations
import inspect
import re
import typing
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Host, Mount, Route
def OpenAPIResponse(self, request: Request) -> Response:
    routes = request.app.routes
    schema = self.get_schema(routes=routes)
    return OpenAPIResponse(schema)