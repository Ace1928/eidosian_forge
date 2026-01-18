from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
class TimeoutTestCase(base.BaseTestCase):

    def setUp(self):
        super(TimeoutTestCase, self).setUp()
        self.messaging_conf = messaging_conffixture.ConfFixture(CONF)
        self.messaging_conf.transport_url = 'fake://'
        self.messaging_conf.response_timeout = 0
        self.useFixture(self.messaging_conf)
        self.addCleanup(rpc.cleanup)
        rpc.init(CONF)
        rpc.TRANSPORT = mock.MagicMock()
        rpc.TRANSPORT._send.side_effect = messaging.MessagingTimeout
        rpc.TRANSPORT.conf.oslo_messaging_metrics.metrics_enabled = False
        target = messaging.Target(version='1.0', topic='testing')
        self.client = rpc.get_client(target)
        self.call_context = mock.Mock()
        self.sleep = mock.patch('time.sleep').start()
        rpc.TRANSPORT.conf.rpc_response_timeout = 10
        rpc.TRANSPORT.conf.rpc_response_max_timeout = 300

    def test_timeout_unaffected_when_explicitly_set(self):
        rpc.TRANSPORT.conf.rpc_response_timeout = 5
        ctx = self.client.prepare(topic='sandwiches', timeout=77)
        with testtools.ExpectedException(messaging.MessagingTimeout):
            ctx.call(self.call_context, 'create_pb_and_j')
        self.assertEqual(5, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['create_pb_and_j'])
        self.assertFalse(self.sleep.called)

    def test_timeout_store_defaults(self):
        self.assertEqual(rpc.TRANSPORT.conf.rpc_response_timeout, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'])
        self.assertEqual(rpc.TRANSPORT.conf.rpc_response_timeout, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_2'])
        rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_2'] = 7000
        self.assertEqual(rpc.TRANSPORT.conf.rpc_response_timeout, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'])
        self.assertEqual(rpc.TRANSPORT.conf.rpc_response_timeout, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_3'])

    def test_method_timeout_sleep(self):
        rpc.TRANSPORT.conf.rpc_response_timeout = 2
        for i in range(100):
            with testtools.ExpectedException(messaging.MessagingTimeout):
                self.client.call(self.call_context, 'method_1')
            self.assertGreaterEqual(self.sleep.call_args_list[0][0][0], 0)
            self.assertLessEqual(self.sleep.call_args_list[0][0][0], 2)
            self.sleep.reset_mock()

    def test_method_timeout_increases_on_timeout_exception(self):
        rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'] = 1
        for i in range(5):
            with testtools.ExpectedException(messaging.MessagingTimeout):
                self.client.call(self.call_context, 'method_1')
        timeouts = [call[1]['timeout'] for call in rpc.TRANSPORT._send.call_args_list]
        self.assertEqual([1, 2, 4, 8, 16], timeouts)

    def test_method_timeout_config_ceiling(self):
        rpc.TRANSPORT.conf.rpc_response_timeout = 10
        for i in range(5):
            with testtools.ExpectedException(messaging.MessagingTimeout):
                self.client.call(self.call_context, 'method_1')
        self.assertEqual(rpc.TRANSPORT.conf.rpc_response_max_timeout, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'])
        with testtools.ExpectedException(messaging.MessagingTimeout):
            self.client.call(self.call_context, 'method_1')
        self.assertEqual(rpc.TRANSPORT.conf.rpc_response_max_timeout, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'])

    def test_timeout_unchanged_on_other_exception(self):
        rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'] = 1
        rpc.TRANSPORT._send.side_effect = ValueError
        with testtools.ExpectedException(ValueError):
            self.client.call(self.call_context, 'method_1')
        rpc.TRANSPORT._send.side_effect = messaging.MessagingTimeout
        with testtools.ExpectedException(messaging.MessagingTimeout):
            self.client.call(self.call_context, 'method_1')
        timeouts = [call[1]['timeout'] for call in rpc.TRANSPORT._send.call_args_list]
        self.assertEqual([1, 1], timeouts)

    def test_timeouts_for_methods_tracked_independently(self):
        rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'] = 1
        rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_2'] = 1
        for method in ('method_1', 'method_1', 'method_2', 'method_1', 'method_2'):
            with testtools.ExpectedException(messaging.MessagingTimeout):
                self.client.call(self.call_context, method)
        timeouts = [call[1]['timeout'] for call in rpc.TRANSPORT._send.call_args_list]
        self.assertEqual([1, 2, 1, 4, 2], timeouts)

    def test_timeouts_for_namespaces_tracked_independently(self):
        rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['ns1.method'] = 1
        rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['ns2.method'] = 1
        for ns in ('ns1', 'ns2'):
            self.client.target.namespace = ns
            for i in range(4):
                with testtools.ExpectedException(messaging.MessagingTimeout):
                    self.client.call(self.call_context, 'method')
        timeouts = [call[1]['timeout'] for call in rpc.TRANSPORT._send.call_args_list]
        self.assertEqual([1, 2, 4, 8, 1, 2, 4, 8], timeouts)

    def test_method_timeout_increases_with_prepare(self):
        rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'] = 1
        ctx = self.client.prepare(version='1.4')
        with testtools.ExpectedException(messaging.MessagingTimeout):
            ctx.call(self.call_context, 'method_1')
        with testtools.ExpectedException(messaging.MessagingTimeout):
            ctx.call(self.call_context, 'method_1')
        timeouts = [call[1]['timeout'] for call in rpc.TRANSPORT._send.call_args_list]
        self.assertEqual([1, 2], timeouts)

    def test_set_max_timeout_caps_all_methods(self):
        rpc.TRANSPORT.conf.rpc_response_timeout = 300
        rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'] = 100
        rpc.BackingOffClient.set_max_timeout(50)
        self.assertEqual(50, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'])
        self.assertEqual(50, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_2'])

    def test_set_max_timeout_retains_lower_timeouts(self):
        rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'] = 10
        rpc.BackingOffClient.set_max_timeout(50)
        self.assertEqual(10, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'])

    def test_set_max_timeout_overrides_default_timeout(self):
        rpc.TRANSPORT.conf.rpc_response_timeout = 10
        self.assertEqual(rpc.TRANSPORT.conf.rpc_response_max_timeout, rpc._BackingOffContextWrapper.get_max_timeout())
        rpc._BackingOffContextWrapper.set_max_timeout(10)
        self.assertEqual(10, rpc._BackingOffContextWrapper.get_max_timeout())