import datetime
import logging
import sys
import uuid
import fixtures
from kombu import connection
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import timeutils
from stevedore import dispatch
from stevedore import extension
import testscenarios
import yaml
import oslo_messaging
from oslo_messaging.notify import _impl_log
from oslo_messaging.notify import _impl_test
from oslo_messaging.notify import messaging
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class TestRoutingNotifier(test_utils.BaseTestCase):

    def setUp(self):
        super(TestRoutingNotifier, self).setUp()
        self.config(driver=['routing'], group='oslo_messaging_notifications')
        transport = oslo_messaging.get_notification_transport(self.conf, url='fake:')
        self.notifier = oslo_messaging.Notifier(transport)
        self.router = self.notifier._driver_mgr['routing'].obj
        self.assertTrue(self.notifier.is_enabled())

    def _fake_extension_manager(self, ext):
        return extension.ExtensionManager.make_test_instance([extension.Extension('test', None, None, ext)])

    def _empty_extension_manager(self):
        return extension.ExtensionManager.make_test_instance([])

    def test_should_load_plugin(self):
        self.router.used_drivers = set(['zoo', 'blah'])
        ext = mock.MagicMock()
        ext.name = 'foo'
        self.assertFalse(self.router._should_load_plugin(ext))
        ext.name = 'zoo'
        self.assertTrue(self.router._should_load_plugin(ext))

    def test_load_notifiers_no_config(self):
        self.router._load_notifiers()
        self.assertEqual({}, self.router.routing_groups)
        self.assertEqual(0, len(self.router.used_drivers))

    def test_load_notifiers_no_extensions(self):
        self.config(routing_config='routing_notifier.yaml', group='oslo_messaging_notifications')
        routing_config = ''
        config_file = mock.MagicMock()
        config_file.return_value = routing_config
        with mock.patch.object(self.router, '_get_notifier_config_file', config_file):
            with mock.patch('stevedore.dispatch.DispatchExtensionManager', return_value=self._empty_extension_manager()):
                with mock.patch('oslo_messaging.notify._impl_routing.LOG') as mylog:
                    self.router._load_notifiers()
                    self.assertFalse(mylog.debug.called)
        self.assertEqual({}, self.router.routing_groups)

    def test_load_notifiers_config(self):
        self.config(routing_config='routing_notifier.yaml', group='oslo_messaging_notifications')
        routing_config = '\ngroup_1:\n   rpc : foo\ngroup_2:\n   rpc : blah\n        '
        config_file = mock.MagicMock()
        config_file.return_value = routing_config
        with mock.patch.object(self.router, '_get_notifier_config_file', config_file):
            with mock.patch('stevedore.dispatch.DispatchExtensionManager', return_value=self._fake_extension_manager(mock.MagicMock())):
                with mock.patch('oslo_messaging.notify._impl_routing.LOG'):
                    self.router._load_notifiers()
                    groups = list(self.router.routing_groups.keys())
                    groups.sort()
                    self.assertEqual(['group_1', 'group_2'], groups)

    def test_get_drivers_for_message_accepted_events(self):
        config = '\ngroup_1:\n   rpc:\n       accepted_events:\n          - foo.*\n          - blah.zoo.*\n          - zip\n        '
        groups = yaml.safe_load(config)
        group = groups['group_1']
        self.assertEqual([], self.router._get_drivers_for_message(group, 'unknown', 'info'))
        self.assertEqual(['rpc'], self.router._get_drivers_for_message(group, 'foo.1', 'info'))
        self.assertEqual([], self.router._get_drivers_for_message(group, 'foo', 'info'))
        self.assertEqual(['rpc'], self.router._get_drivers_for_message(group, 'blah.zoo.zing', 'info'))

    def test_get_drivers_for_message_accepted_priorities(self):
        config = '\ngroup_1:\n   rpc:\n       accepted_priorities:\n          - info\n          - error\n        '
        groups = yaml.safe_load(config)
        group = groups['group_1']
        self.assertEqual([], self.router._get_drivers_for_message(group, None, 'unknown'))
        self.assertEqual(['rpc'], self.router._get_drivers_for_message(group, None, 'info'))
        self.assertEqual(['rpc'], self.router._get_drivers_for_message(group, None, 'error'))

    def test_get_drivers_for_message_both(self):
        config = '\ngroup_1:\n   rpc:\n       accepted_priorities:\n          - info\n       accepted_events:\n          - foo.*\n   driver_1:\n       accepted_priorities:\n          - info\n   driver_2:\n      accepted_events:\n          - foo.*\n        '
        groups = yaml.safe_load(config)
        group = groups['group_1']
        self.assertEqual(['driver_2'], self.router._get_drivers_for_message(group, 'foo.blah', 'unknown'))
        self.assertEqual(['driver_1'], self.router._get_drivers_for_message(group, 'unknown', 'info'))
        x = self.router._get_drivers_for_message(group, 'foo.blah', 'info')
        x.sort()
        self.assertEqual(['driver_1', 'driver_2', 'rpc'], x)

    def test_filter_func(self):
        ext = mock.MagicMock()
        ext.name = 'rpc'
        self.assertTrue(self.router._filter_func(ext, {}, {}, 'info', None, ['foo', 'rpc']))
        self.assertFalse(self.router._filter_func(ext, {}, {}, 'info', None, ['foo']))

    def test_notify(self):
        self.router.routing_groups = {'group_1': None, 'group_2': None}
        drivers_mock = mock.MagicMock()
        drivers_mock.side_effect = [['rpc'], ['foo']]
        with mock.patch.object(self.router, 'plugin_manager') as pm:
            with mock.patch.object(self.router, '_get_drivers_for_message', drivers_mock):
                self.notifier.info(test_utils.TestContext(), 'my_event', {})
                self.assertEqual(sorted(['rpc', 'foo']), sorted(pm.map.call_args[0][6]))

    def test_notify_filtered(self):
        self.config(routing_config='routing_notifier.yaml', group='oslo_messaging_notifications')
        routing_config = '\ngroup_1:\n    rpc:\n        accepted_events:\n          - my_event\n    rpc2:\n        accepted_priorities:\n          - info\n    bar:\n        accepted_events:\n            - nothing\n        '
        config_file = mock.MagicMock()
        config_file.return_value = routing_config
        rpc_driver = mock.Mock()
        rpc2_driver = mock.Mock()
        bar_driver = mock.Mock()
        pm = dispatch.DispatchExtensionManager.make_test_instance([extension.Extension('rpc', None, None, rpc_driver), extension.Extension('rpc2', None, None, rpc2_driver), extension.Extension('bar', None, None, bar_driver)])
        with mock.patch.object(self.router, '_get_notifier_config_file', config_file):
            with mock.patch('stevedore.dispatch.DispatchExtensionManager', return_value=pm):
                with mock.patch('oslo_messaging.notify._impl_routing.LOG'):
                    cxt = test_utils.TestContext()
                    self.notifier.info(cxt, 'my_event', {})
                    self.assertFalse(bar_driver.info.called)
                    rpc_driver.notify.assert_called_once_with(cxt, mock.ANY, 'INFO', -1)
                    rpc2_driver.notify.assert_called_once_with(cxt, mock.ANY, 'INFO', -1)