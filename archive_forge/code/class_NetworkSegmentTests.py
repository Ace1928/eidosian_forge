import uuid
from openstackclient.tests.functional.network.v2 import common
class NetworkSegmentTests(common.NetworkTests):
    """Functional tests for network segment"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if cls.haz_network:
            cls.NETWORK_NAME = uuid.uuid4().hex
            cls.PHYSICAL_NETWORK_NAME = uuid.uuid4().hex
            cmd_output = cls.openstack('network create ' + cls.NETWORK_NAME, parse_output=True)
            cls.NETWORK_ID = cmd_output['id']

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.haz_network:
                raw_output = cls.openstack('network delete ' + cls.NETWORK_NAME)
                cls.assertOutput('', raw_output)
        finally:
            super().tearDownClass()

    def setUp(self):
        super().setUp()
        if not self.is_extension_enabled('segment'):
            self.skipTest('No segment extension present')

    def test_network_segment_create_delete(self):
        name = uuid.uuid4().hex
        json_output = self.openstack(' network segment create ' + '--network ' + self.NETWORK_ID + ' ' + '--network-type geneve ' + '--segment 2055 ' + name, parse_output=True)
        self.assertEqual(name, json_output['name'])
        raw_output = self.openstack('network segment delete ' + name)
        self.assertOutput('', raw_output)

    def test_network_segment_list(self):
        name = uuid.uuid4().hex
        json_output = self.openstack(' network segment create ' + '--network ' + self.NETWORK_ID + ' ' + '--network-type geneve ' + '--segment 2055 ' + name, parse_output=True)
        network_segment_id = json_output.get('id')
        network_segment_name = json_output.get('name')
        self.addCleanup(self.openstack, 'network segment delete ' + network_segment_id)
        self.assertEqual(name, json_output['name'])
        json_output = self.openstack('network segment list', parse_output=True)
        item_map = {item.get('ID'): item.get('Name') for item in json_output}
        self.assertIn(network_segment_id, item_map.keys())
        self.assertIn(network_segment_name, item_map.values())

    def test_network_segment_set_show(self):
        name = uuid.uuid4().hex
        json_output = self.openstack(' network segment create ' + '--network ' + self.NETWORK_ID + ' ' + '--network-type geneve ' + '--segment 2055 ' + name, parse_output=True)
        self.addCleanup(self.openstack, 'network segment delete ' + name)
        if self.is_extension_enabled('standard-attr-segment'):
            self.assertEqual('', json_output['description'])
        else:
            self.assertIsNone(json_output['description'])
        new_description = 'new_description'
        cmd_output = self.openstack('network segment set ' + '--description ' + new_description + ' ' + name)
        self.assertOutput('', cmd_output)
        json_output = self.openstack('network segment show ' + name, parse_output=True)
        self.assertEqual(new_description, json_output['description'])