from __future__ import absolute_import
import asyncio
import functools
import aiohttp  # type: ignore
import six
import urllib3  # type: ignore
from google.auth import exceptions
from google.auth import transport
from google.auth.transport import requests
class _Response(transport.Response):
    """
    Requests transport response adapter.

    Args:
        response (requests.Response): The raw Requests response.
    """

    def __init__(self, response):
        self._response = response

    @property
    def status(self):
        return self._response.status

    @property
    def headers(self):
        return self._response.headers

    @property
    def data(self):
        return self._response.content