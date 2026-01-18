from unittest import mock
from keystoneauth1 import exceptions as keystone_exceptions
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.tests import common
class KeystoneClientPluginServiceTest(common.HeatTestCase):
    sample_uuid = '477e8273-60a7-4c41-b683-fdb0bc7cd152'
    sample_name = 'sample_service'

    def _get_mock_service(self):
        srv = mock.MagicMock()
        srv.id = self.sample_uuid
        srv.name = self.sample_name
        return srv

    def setUp(self):
        super(KeystoneClientPluginServiceTest, self).setUp()
        self._client = mock.MagicMock()

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_service_id(self, client_keystone):
        self._client.client.services.get.return_value = self._get_mock_service()
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        self.assertEqual(self.sample_uuid, client_plugin.get_service_id(self.sample_uuid))
        self._client.client.services.get.assert_called_once_with(self.sample_uuid)

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_service_id_with_name(self, client_keystone):
        self._client.client.services.get.side_effect = keystone_exceptions.NotFound
        self._client.client.services.list.return_value = [self._get_mock_service()]
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        self.assertEqual(self.sample_uuid, client_plugin.get_service_id(self.sample_name))
        self.assertRaises(keystone_exceptions.NotFound, self._client.client.services.get, self.sample_name)
        self._client.client.services.list.assert_called_once_with(name=self.sample_name)

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_service_id_with_name_conflict(self, client_keystone):
        self._client.client.services.get.side_effect = keystone_exceptions.NotFound
        self._client.client.services.list.return_value = [self._get_mock_service(), self._get_mock_service()]
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        ex = self.assertRaises(exception.KeystoneServiceNameConflict, client_plugin.get_service_id, self.sample_name)
        msg = 'Keystone has more than one service with same name %s. Please use service id instead of name' % self.sample_name
        self.assertEqual(msg, str(ex))
        self.assertRaises(keystone_exceptions.NotFound, self._client.client.services.get, self.sample_name)
        self._client.client.services.list.assert_called_once_with(name=self.sample_name)

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_service_id_not_found(self, client_keystone):
        self._client.client.services.get.side_effect = keystone_exceptions.NotFound
        self._client.client.services.list.return_value = []
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        ex = self.assertRaises(exception.EntityNotFound, client_plugin.get_service_id, self.sample_name)
        msg = 'The KeystoneService (%(name)s) could not be found.' % {'name': self.sample_name}
        self.assertEqual(msg, str(ex))
        self.assertRaises(keystone_exceptions.NotFound, self._client.client.services.get, self.sample_name)
        self._client.client.services.list.assert_called_once_with(name=self.sample_name)