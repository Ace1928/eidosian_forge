import codecs
import gzip
import os
import six.moves.urllib.request as urllib_request
import tempfile
import unittest
from apitools.gen import util
from mock import patch
class MockRequestResponse:
    """Mocks the behavior of urllib.response."""

    class MockRequestEncoding:

        def __init__(self, encoding):
            self.encoding = encoding

        def get(self, _):
            return self.encoding

    def __init__(self, content, encoding):
        self.content = content
        self.encoding = MockRequestResponse.MockRequestEncoding(encoding)

    def info(self):
        return self.encoding

    def read(self):
        return self.content