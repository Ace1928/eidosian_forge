from os_brick.initiator.connectors import gpfs
from os_brick.tests.initiator.connectors import test_local
class GPFSConnectorTestCase(test_local.LocalConnectorTestCase):

    def setUp(self):
        super(GPFSConnectorTestCase, self).setUp()
        self.connection_properties = {'name': 'foo', 'device_path': '/tmp/bar'}
        self.connector = gpfs.GPFSConnector(None)

    def test_connect_volume(self):
        cprops = self.connection_properties
        dev_info = self.connector.connect_volume(cprops)
        self.assertEqual(dev_info['type'], 'gpfs')
        self.assertEqual(dev_info['path'], cprops['device_path'])

    def test_connect_volume_with_invalid_connection_data(self):
        cprops = {}
        self.assertRaises(ValueError, self.connector.connect_volume, cprops)