import testscenarios
import time
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class TestDispatcherWithPingEndpoint(test_utils.BaseTestCase):

    def test_dispatcher_with_ping(self):
        self.config(rpc_ping_enabled=True)
        dispatcher = oslo_messaging.RPCDispatcher([], None, None)
        incoming = mock.Mock(ctxt={}, message=dict(method='oslo_rpc_server_ping'), client_timeout=0)
        res = dispatcher.dispatch(incoming)
        self.assertEqual('pong', res)

    def test_dispatcher_with_ping_already_used(self):

        class MockEndpoint(object):

            def oslo_rpc_server_ping(self, ctxt, **kwargs):
                return 'not_pong'
        mockEndpoint = MockEndpoint()
        self.config(rpc_ping_enabled=True)
        dispatcher = oslo_messaging.RPCDispatcher([mockEndpoint], None, None)
        incoming = mock.Mock(ctxt={}, message=dict(method='oslo_rpc_server_ping'), client_timeout=0)
        res = dispatcher.dispatch(incoming)
        self.assertEqual('not_pong', res)