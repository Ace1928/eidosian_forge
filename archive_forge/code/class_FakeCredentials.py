import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
class FakeCredentials(object):

    def __init__(self):
        self.num_refreshes = 0

    def refresh(self, _):
        self.num_refreshes += 1