import warnings
import pytest
from urllib3.connection import HTTPConnection
from urllib3.packages.six.moves import http_cookiejar, urllib
from urllib3.response import HTTPResponse
class TestVersionCompatibility(object):

    def test_connection_strict(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            HTTPConnection('localhost', 12345, strict=True)
            if w:
                pytest.fail('HTTPConnection raised warning on strict=True: %r' % w[0].message)

    def test_connection_source_address(self):
        try:
            HTTPConnection('localhost', 12345, source_address='127.0.0.1')
        except TypeError as e:
            pytest.fail('HTTPConnection raised TypeError on source_address: %r' % e)