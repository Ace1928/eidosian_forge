from unittest import mock
from keystoneauth1 import exceptions as keystone_exceptions
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.tests import common
class KeystoneClientPluginRegionTest(common.HeatTestCase):
    sample_uuid = '477e8273-60a7-4c41-b683-fdb0bc7cd152'
    sample_name = 'sample_region'

    def _get_mock_region(self):
        region = mock.MagicMock()
        region.id = self.sample_uuid
        region.name = self.sample_name
        return region

    def setUp(self):
        super(KeystoneClientPluginRegionTest, self).setUp()
        self._client = mock.MagicMock()

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_region_id(self, client_keystone):
        self._client.client.regions.get.return_value = self._get_mock_region()
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        self.assertEqual(self.sample_uuid, client_plugin.get_region_id(self.sample_uuid))
        self._client.client.regions.get.assert_called_once_with(self.sample_uuid)

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_region_id_not_found(self, client_keystone):
        self._client.client.regions.get.side_effect = keystone_exceptions.NotFound
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        ex = self.assertRaises(exception.EntityNotFound, client_plugin.get_region_id, self.sample_name)
        msg = 'The KeystoneRegion (%(name)s) could not be found.' % {'name': self.sample_name}
        self.assertEqual(msg, str(ex))