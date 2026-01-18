import io
import os
import threading
import time
import uuid
from functools import lru_cache
from http import HTTPStatus
from typing import Callable, Tuple, Type, Union
import requests
from requests import Response
from requests.adapters import HTTPAdapter
from requests.models import PreparedRequest
from .. import constants
from . import logging
from ._typing import HTTP_METHOD_T
class UniqueRequestIdAdapter(HTTPAdapter):
    X_AMZN_TRACE_ID = 'X-Amzn-Trace-Id'

    def add_headers(self, request, **kwargs):
        super().add_headers(request, **kwargs)
        if X_AMZN_TRACE_ID not in request.headers:
            request.headers[X_AMZN_TRACE_ID] = request.headers.get(X_REQUEST_ID) or str(uuid.uuid4())
        has_token = str(request.headers.get('authorization', '')).startswith('Bearer hf_')
        logger.debug(f'Request {request.headers[X_AMZN_TRACE_ID]}: {request.method} {request.url} (authenticated: {has_token})')

    def send(self, request: PreparedRequest, *args, **kwargs) -> Response:
        """Catch any RequestException to append request id to the error message for debugging."""
        try:
            return super().send(request, *args, **kwargs)
        except requests.RequestException as e:
            request_id = request.headers.get(X_AMZN_TRACE_ID)
            if request_id is not None:
                e.args = (*e.args, f'(Request ID: {request_id})')
            raise