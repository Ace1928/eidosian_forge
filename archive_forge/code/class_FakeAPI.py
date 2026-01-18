import copy
import datetime
import io
import os
from oslo_serialization import jsonutils
import queue
import sys
import fixtures
import testtools
from magnumclient.common import httpclient as http
from magnumclient import shell
class FakeAPI(object):

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def _request(self, method, url, headers=None, body=None):
        call = (method, url, headers or {}, body)
        self.calls.append(call)
        return self.responses[url][method]

    def raw_request(self, *args, **kwargs):
        response = self._request(*args, **kwargs)
        body_iter = http.ResponseBodyIterator(io.StringIO(response[1]))
        return (FakeResponse(response[0]), body_iter)

    def json_request(self, *args, **kwargs):
        response = self._request(*args, **kwargs)
        return (FakeResponse(response[0]), response[1])