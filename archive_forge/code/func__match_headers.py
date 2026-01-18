import urllib.parse
import weakref
from requests.adapters import BaseAdapter
from requests.utils import requote_uri
from requests_mock import exceptions
from requests_mock.request import _RequestObjectProxy
from requests_mock.response import _MatcherResponse
import logging
def _match_headers(self, request):
    for k, vals in self._request_headers.items():
        try:
            header = request.headers[k]
        except KeyError:
            if not isinstance(k, str):
                return False
            try:
                header = request.headers[k.encode('utf-8')]
            except KeyError:
                return False
        if header != vals:
            return False
    return True