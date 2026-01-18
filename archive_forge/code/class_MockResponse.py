import httplib
import pytest
import StringIO
from mock import patch
from ..test_no_ssl import TestWithoutSSL
class MockResponse(object):

    def __init__(self, content, status_code, content_was_truncated, final_url, headers):
        self.content = content
        self.status_code = status_code
        self.content_was_truncated = content_was_truncated
        self.final_url = final_url
        self.header_msg = httplib.HTTPMessage(StringIO.StringIO(''.join(['%s: %s\n' % (k, v) for k, v in headers.iteritems()] + ['\n'])))
        self.headers = headers