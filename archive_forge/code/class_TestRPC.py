from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
class TestRPC(base.BaseTestCase):

    def setUp(self):
        super(TestRPC, self).setUp()
        self.useFixture(fixture.RPCFixture())

    @mock.patch.object(rpc, 'RequestContextSerializer')
    @mock.patch.object(messaging, 'get_rpc_transport')
    @mock.patch.object(messaging, 'get_notification_transport')
    def test_init(self, mock_noti_trans, mock_trans, mock_ser):
        transport = mock.Mock()
        noti_transport = mock.Mock()
        conf = mock.Mock()
        mock_trans.return_value = transport
        mock_noti_trans.return_value = noti_transport
        rpc.init(conf, rpc_ext_mods=['foo'])
        expected_mods = list(set(['foo'] + rpc._DFT_EXMODS))
        mock_trans.assert_called_once_with(conf, allowed_remote_exmods=expected_mods)
        mock_noti_trans.assert_called_once_with(conf, allowed_remote_exmods=expected_mods)
        self.assertIsNotNone(rpc.TRANSPORT)
        self.assertIsNotNone(rpc.NOTIFICATION_TRANSPORT)

    def test_cleanup_transport_null(self):
        rpc.NOTIFICATION_TRANSPORT = mock.Mock()
        rpc.TRANSPORT = None
        self.assertRaises(AssertionError, rpc.cleanup)
        rpc.TRANSPORT = mock.Mock()

    def test_cleanup_notification_transport_null(self):
        rpc.TRANSPORT = mock.Mock()
        rpc.NOTIFICATION_TRANSPORT = None
        self.assertRaises(AssertionError, rpc.cleanup)
        rpc.NOTIFICATION_TRANSPORT = mock.Mock()

    def test_cleanup(self):
        rpc.NOTIFICATION_TRANSPORT = mock.Mock()
        rpc.TRANSPORT = mock.Mock()
        trans_cleanup = mock.Mock()
        not_trans_cleanup = mock.Mock()
        rpc.TRANSPORT.cleanup = trans_cleanup
        rpc.NOTIFICATION_TRANSPORT.cleanup = not_trans_cleanup
        rpc.cleanup()
        trans_cleanup.assert_called_once_with()
        not_trans_cleanup.assert_called_once_with()
        self.assertIsNone(rpc.TRANSPORT)
        self.assertIsNone(rpc.NOTIFICATION_TRANSPORT)
        rpc.TRANSPORT = mock.Mock()
        rpc.NOTIFICATION_TRANSPORT = mock.Mock()

    @mock.patch.object(rpc, 'RequestContextSerializer')
    @mock.patch.object(messaging, 'get_rpc_client')
    def test_get_client(self, mock_get, mock_ser):
        rpc.TRANSPORT = mock.Mock()
        tgt = mock.Mock()
        ser = mock.Mock()
        mock_get.return_value = 'client'
        mock_ser.return_value = ser
        client = rpc.get_client(tgt, version_cap='1.0', serializer='foo')
        mock_ser.assert_called_once_with('foo')
        mock_get.assert_called_once_with(rpc.TRANSPORT, tgt, version_cap='1.0', serializer=ser, client_cls=rpc.BackingOffClient)
        self.assertEqual('client', client)

    @mock.patch.object(rpc, 'RequestContextSerializer')
    @mock.patch.object(messaging, 'get_rpc_server')
    def test_get_server(self, mock_get, mock_ser):
        ser = mock.Mock()
        tgt = mock.Mock()
        ends = mock.Mock()
        mock_ser.return_value = ser
        mock_get.return_value = 'server'
        server = rpc.get_server(tgt, ends, serializer='foo')
        mock_ser.assert_called_once_with('foo')
        mock_get.assert_called_once_with(rpc.TRANSPORT, tgt, ends, 'eventlet', ser)
        self.assertEqual('server', server)

    def test_get_notifier(self):
        mock_notifier = mock.Mock(return_value=None)
        messaging.Notifier.__init__ = mock_notifier
        rpc.get_notifier('service', publisher_id='foo')
        mock_notifier.assert_called_once_with(mock.ANY, serializer=mock.ANY, publisher_id='foo')

    def test_get_notifier_null_publisher(self):
        mock_notifier = mock.Mock(return_value=None)
        messaging.Notifier.__init__ = mock_notifier
        rpc.get_notifier('service', host='bar')
        mock_notifier.assert_called_once_with(mock.ANY, serializer=mock.ANY, publisher_id='service.bar')