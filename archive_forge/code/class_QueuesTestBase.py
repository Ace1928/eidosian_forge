from unittest import mock
from oslo_utils import netutils
from zaqarclient.queues import client
from zaqarclient.tests import base
from zaqarclient.tests.transport import dummy
class QueuesTestBase(base.TestBase):
    transport_cls = dummy.DummyTransport
    url = 'http://%s:8888' % MY_IP
    version = 1

    def setUp(self):
        super(QueuesTestBase, self).setUp()
        self.transport = self.transport_cls(self.conf)
        self.client = client.Client(self.url, self.version, self.conf)
        mocked_transport = mock.Mock(return_value=self.transport)
        self.client._get_transport = mocked_transport
        self.queue = self.client.queue(1, auto_create=False)