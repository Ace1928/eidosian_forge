import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def _MakeSampleRequest(self, url, name):
    return http_wrapper.Request(url, 'POST', {'content-type': 'multipart/mixed; boundary="None"', 'content-length': 80}, '{0} {1}'.format(name, 'x' * (79 - len(name))))