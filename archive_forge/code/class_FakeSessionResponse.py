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
class FakeSessionResponse(object):

    def __init__(self, headers, content=None, status_code=None):
        self.headers = headers
        self.content = content
        self.status_code = status_code

    def json(self):
        if self.content is not None:
            return jsonutils.loads(self.content)
        else:
            return {}