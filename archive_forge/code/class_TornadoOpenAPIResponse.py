from __future__ import annotations
import json
import os
import sys
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import tornado.httpclient
import tornado.web
from openapi_core import V30RequestValidator, V30ResponseValidator
from openapi_core.spec.paths import Spec
from openapi_core.validation.request.datatypes import RequestParameters
from tornado.httpclient import HTTPRequest, HTTPResponse
from werkzeug.datastructures import Headers, ImmutableMultiDict
from jupyterlab_server.spec import get_openapi_spec
class TornadoOpenAPIResponse:
    """A tornado open API response."""

    def __init__(self, response: HTTPResponse):
        """Initialize the response."""
        self.response = response

    @property
    def data(self) -> bytes | None:
        if not isinstance(self.response.body, bytes):
            msg = 'Response body is invalid'
            raise AssertionError(msg)
        return self.response.body

    @property
    def status_code(self) -> int:
        return int(self.response.code)

    @property
    def content_type(self) -> str:
        return 'application/json'

    @property
    def mimetype(self) -> str:
        return str(self.response.headers.get('Content-Type', 'application/json'))

    @property
    def headers(self) -> Headers:
        return Headers(dict(self.response.headers))