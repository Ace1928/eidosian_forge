import copy
import io
import json
import testtools
from urllib import parse
from glanceclient.v2 import schemas
class RawRequest(object):

    def __init__(self, headers, body=None, version=1.0, status=200, reason='Ok'):
        """A crafted request object used for testing.

        :param headers: dict representing HTTP response headers
        :param body: file-like object
        :param version: HTTP Version
        :param status: Response status code
        :param reason: Status code related message.
        """
        self.body = body
        self.status = status
        self.reason = reason
        self.version = version
        self.headers = headers

    def getheaders(self):
        return copy.deepcopy(self.headers).items()

    def getheader(self, key, default):
        return self.headers.get(key, default)

    def read(self, amt):
        return self.body.read(amt)