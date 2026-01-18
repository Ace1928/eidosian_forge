import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
class TestSOCKSProxyManager(object):

    def test_invalid_socks_version_is_valueerror(self):
        with pytest.raises(ValueError) as e:
            socks.SOCKSProxyManager(proxy_url='http://example.org')
        assert 'Unable to determine SOCKS version' in e.value.args[0]