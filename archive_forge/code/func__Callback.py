import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def _Callback(response, exception):
    self.assertEqual({'status': '200'}, response.info)
    self.assertEqual('content', response.content)
    self.assertEqual(desired_url, response.request_url)
    self.assertIsNone(exception)
    callback_was_called.append(1)