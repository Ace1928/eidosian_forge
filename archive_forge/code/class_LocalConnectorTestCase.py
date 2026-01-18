from os_brick.initiator.connectors import local
from os_brick.tests.initiator import test_connector
class LocalConnectorTestCase(test_connector.ConnectorTestCase):

    def setUp(self):
        super(LocalConnectorTestCase, self).setUp()
        self.connection_properties = {'name': 'foo', 'device_path': '/tmp/bar'}
        self.connector = local.LocalConnector(None)

    def test_get_connector_properties(self):
        props = local.LocalConnector.get_connector_properties('sudo', multipath=True, enforce_multipath=True)
        expected_props = {}
        self.assertEqual(expected_props, props)

    def test_get_search_path(self):
        actual = self.connector.get_search_path()
        self.assertIsNone(actual)

    def test_get_volume_paths(self):
        expected = [self.connection_properties['device_path']]
        actual = self.connector.get_volume_paths(self.connection_properties)
        self.assertEqual(expected, actual)

    def test_connect_volume(self):
        cprops = self.connection_properties
        dev_info = self.connector.connect_volume(cprops)
        self.assertEqual(dev_info['type'], 'local')
        self.assertEqual(dev_info['path'], cprops['device_path'])

    def test_connect_volume_with_invalid_connection_data(self):
        cprops = {}
        self.assertRaises(ValueError, self.connector.connect_volume, cprops)

    def test_extend_volume(self):
        self.assertRaises(NotImplementedError, self.connector.extend_volume, self.connection_properties)