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
class FakeSession(object):

    def __init__(self, headers, content=None, status_code=None):
        self.headers = headers
        self.content = content
        self.status_code = status_code

    def request(self, url, method, **kwargs):
        return FakeSessionResponse(self.headers, self.content, self.status_code)