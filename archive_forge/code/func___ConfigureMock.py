import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def __ConfigureMock(self, mock_request, expected_request, response):
    if isinstance(response, list):
        response = list(response)

    def CheckRequest(_, request, **unused_kwds):
        self.assertUrlEqual(expected_request.url, request.url)
        self.assertEqual(expected_request.http_method, request.http_method)
        if isinstance(response, list):
            return response.pop(0)
        return response
    mock_request.side_effect = CheckRequest