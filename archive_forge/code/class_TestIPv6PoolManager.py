import json
from test import LONG_TIMEOUT
import pytest
from dummyserver.server import HAS_IPV6
from dummyserver.testcase import HTTPDummyServerTestCase, IPv6HTTPDummyServerTestCase
from urllib3.connectionpool import port_by_scheme
from urllib3.exceptions import MaxRetryError, URLSchemeUnknown
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
@pytest.mark.skipif(not HAS_IPV6, reason='IPv6 is not supported on this system')
class TestIPv6PoolManager(IPv6HTTPDummyServerTestCase):

    @classmethod
    def setup_class(cls):
        super(TestIPv6PoolManager, cls).setup_class()
        cls.base_url = 'http://[%s]:%d' % (cls.host, cls.port)

    def test_ipv6(self):
        with PoolManager() as http:
            http.request('GET', self.base_url)