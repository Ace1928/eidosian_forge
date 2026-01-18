from __future__ import annotations
import inspect
import re
import typing
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Host, Mount, Route
class OpenAPIResponse(Response):
    media_type = 'application/vnd.oai.openapi'

    def render(self, content: typing.Any) -> bytes:
        assert yaml is not None, '`pyyaml` must be installed to use OpenAPIResponse.'
        assert isinstance(content, dict), 'The schema passed to OpenAPIResponse should be a dictionary.'
        return yaml.dump(content, default_flow_style=False).encode('utf-8')