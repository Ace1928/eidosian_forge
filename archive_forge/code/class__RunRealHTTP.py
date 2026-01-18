import urllib.parse
import weakref
from requests.adapters import BaseAdapter
from requests.utils import requote_uri
from requests_mock import exceptions
from requests_mock.request import _RequestObjectProxy
from requests_mock.response import _MatcherResponse
import logging
class _RunRealHTTP(Exception):
    """A fake exception to jump out of mocking and allow a real request.

    This exception is caught at the mocker level and allows it to execute this
    request through the real requests mechanism rather than the mocker.

    It should never be exposed to a user.
    """