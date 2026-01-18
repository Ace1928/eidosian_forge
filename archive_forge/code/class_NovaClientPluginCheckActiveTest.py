import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class NovaClientPluginCheckActiveTest(NovaClientPluginTestCase):
    scenarios = [('active', dict(status='ACTIVE', e_raise=False)), ('deferred', dict(status='BUILD', e_raise=False)), ('error', dict(status='ERROR', e_raise=exception.ResourceInError)), ('unknown', dict(status='VIKINGS!', e_raise=exception.ResourceUnknownStatus))]

    def setUp(self):
        super(NovaClientPluginCheckActiveTest, self).setUp()
        self.server = mock.Mock()
        self.server.id = '1234'
        self.server.status = self.status
        self.r_mock = self.patchobject(self.nova_plugin, 'refresh_server', return_value=None)
        self.f_mock = self.patchobject(self.nova_plugin, 'fetch_server', return_value=self.server)

    def test_check_active_with_object(self):
        if self.e_raise:
            self.assertRaises(self.e_raise, self.nova_plugin._check_active, self.server)
            self.r_mock.assert_called_once_with(self.server)
        elif self.status in self.nova_plugin.deferred_server_statuses:
            self.assertFalse(self.nova_plugin._check_active(self.server))
            self.r_mock.assert_called_once_with(self.server)
        else:
            self.assertTrue(self.nova_plugin._check_active(self.server))
            self.assertEqual(0, self.r_mock.call_count)
        self.assertEqual(0, self.f_mock.call_count)

    def test_check_active_with_string(self):
        if self.e_raise:
            self.assertRaises(self.e_raise, self.nova_plugin._check_active, self.server.id)
        elif self.status in self.nova_plugin.deferred_server_statuses:
            self.assertFalse(self.nova_plugin._check_active(self.server.id))
        else:
            self.assertTrue(self.nova_plugin._check_active(self.server.id))
        self.f_mock.assert_called_once_with(self.server.id)
        self.assertEqual(0, self.r_mock.call_count)

    def test_check_active_with_string_unavailable(self):
        self.f_mock.return_value = None
        self.assertFalse(self.nova_plugin._check_active(self.server.id))
        self.f_mock.assert_called_once_with(self.server.id)
        self.assertEqual(0, self.r_mock.call_count)