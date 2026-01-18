import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def _MakeResponse(self, number_of_parts):
    return http_wrapper.Response(info={'status': '200', 'content-type': 'multipart/mixed; boundary="boundary"'}, content='--boundary\n' + '--boundary\n'.join((textwrap.dedent('                    content-type: text/plain\n                    content-id: <id+{0}>\n\n                    HTTP/1.1 200 OK\n                    response {0} content\n\n                    ').format(i) for i in range(number_of_parts))) + '--boundary--', request_url=None)