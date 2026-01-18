import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def _DoTestConvertInvalidHeaderToId(self, invalid_header):
    batch_request = batch.BatchHttpRequest('https://www.example.com')
    self.assertRaises(exceptions.BatchError, batch_request._ConvertHeaderToId, invalid_header)