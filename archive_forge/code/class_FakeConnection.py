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
class FakeConnection(object):

    def __init__(self, response=None, **kwargs):
        self._response = queue.Queue()
        self._response.put(response)
        self._last_request = None
        self._exc = kwargs['exc'] if 'exc' in kwargs else None
        if 'redirect_resp' in kwargs:
            self._response.put(kwargs['redirect_resp'])

    def request(self, method, conn_url, **kwargs):
        self._last_request = (method, conn_url, kwargs)
        if self._exc:
            raise self._exc

    def setresponse(self, response):
        self._response = response

    def getresponse(self):
        return self._response.get()