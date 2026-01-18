import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
class FakeService(object):
    """A service for testing."""

    def GetMethodConfig(self, _):
        return {}

    def GetUploadConfig(self, _):
        return {}

    def PrepareHttpRequest(self, method_config, request, global_params, upload_config):
        return global_params['desired_request']

    def ProcessHttpResponse(self, _, http_response):
        return http_response