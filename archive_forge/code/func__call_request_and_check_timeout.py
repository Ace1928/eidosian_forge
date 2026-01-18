from keystoneauth1.identity import generic
from keystoneauth1 import session as keystone_session
from unittest.mock import Mock
from designateclient.tests import v2
from designateclient.v2.client import Client
def _call_request_and_check_timeout(self, client, timeout):
    """call the mocked _send_request() and check if the timeout was set
        """
    client.limits.get()
    self.assertTrue(self.mock_send_request.called)
    kw = self.mock_send_request.call_args[1]
    if timeout is None:
        self.assertNotIn('timeout', kw)
    else:
        self.assertEqual(timeout, kw['timeout'])