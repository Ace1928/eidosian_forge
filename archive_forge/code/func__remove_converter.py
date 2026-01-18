from __future__ import annotations
import inspect
import re
import typing
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Host, Mount, Route
def _remove_converter(self, path: str) -> str:
    """
        Remove the converter from the path.
        For example, a route like this:
            Route("/users/{id:int}", endpoint=get_user, methods=["GET"])
        Should be represented as `/users/{id}` in the OpenAPI schema.
        """
    return re.sub(':\\w+}', '}', path)