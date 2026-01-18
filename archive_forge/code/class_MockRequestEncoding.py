import codecs
import gzip
import os
import six.moves.urllib.request as urllib_request
import tempfile
import unittest
from apitools.gen import util
from mock import patch
class MockRequestEncoding:

    def __init__(self, encoding):
        self.encoding = encoding

    def get(self, _):
        return self.encoding