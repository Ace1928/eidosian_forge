from openstackclient.tests.functional.network.v2 import common
class NetworkFlavorProfileTests(common.NetworkTests):
    """Functional tests for network flavor profile"""
    DESCRIPTION = 'fakedescription'
    METAINFO = 'Extrainfo'

    def test_network_flavor_profile_create(self):
        json_output = self.openstack('network flavor profile create ' + '--description ' + self.DESCRIPTION + ' ' + '--enable --metainfo ' + self.METAINFO, parse_output=True)
        ID = json_output.get('id')
        self.assertIsNotNone(ID)
        self.assertTrue(json_output.get('enabled'))
        self.assertEqual('fakedescription', json_output.get('description'))
        self.assertEqual('Extrainfo', json_output.get('meta_info'))
        raw_output = self.openstack('network flavor profile delete ' + ID)
        self.assertOutput('', raw_output)

    def test_network_flavor_profile_list(self):
        json_output = self.openstack('network flavor profile create ' + '--description ' + self.DESCRIPTION + ' ' + '--enable ' + '--metainfo ' + self.METAINFO, parse_output=True)
        ID1 = json_output.get('id')
        self.assertIsNotNone(ID1)
        self.assertTrue(json_output.get('enabled'))
        self.assertEqual('fakedescription', json_output.get('description'))
        self.assertEqual('Extrainfo', json_output.get('meta_info'))
        json_output = self.openstack('network flavor profile create ' + '--description ' + self.DESCRIPTION + ' ' + '--disable ' + '--metainfo ' + self.METAINFO, parse_output=True)
        ID2 = json_output.get('id')
        self.assertIsNotNone(ID2)
        self.assertFalse(json_output.get('enabled'))
        self.assertEqual('fakedescription', json_output.get('description'))
        self.assertEqual('Extrainfo', json_output.get('meta_info'))
        json_output = self.openstack('network flavor profile list', parse_output=True)
        self.assertIsNotNone(json_output)
        id_list = [item.get('ID') for item in json_output]
        self.assertIn(ID1, id_list)
        self.assertIn(ID2, id_list)
        raw_output = self.openstack('network flavor profile delete ' + ID1 + ' ' + ID2)
        self.assertOutput('', raw_output)

    def test_network_flavor_profile_set(self):
        json_output_1 = self.openstack('network flavor profile create ' + '--description ' + self.DESCRIPTION + ' ' + '--enable ' + '--metainfo ' + self.METAINFO, parse_output=True)
        ID = json_output_1.get('id')
        self.assertIsNotNone(ID)
        self.assertTrue(json_output_1.get('enabled'))
        self.assertEqual('fakedescription', json_output_1.get('description'))
        self.assertEqual('Extrainfo', json_output_1.get('meta_info'))
        self.openstack('network flavor profile set --disable ' + ID)
        json_output = self.openstack('network flavor profile show ' + ID, parse_output=True)
        self.assertFalse(json_output.get('enabled'))
        self.assertEqual('fakedescription', json_output.get('description'))
        self.assertEqual('Extrainfo', json_output.get('meta_info'))
        raw_output = self.openstack('network flavor profile delete ' + ID)
        self.assertOutput('', raw_output)

    def test_network_flavor_profile_show(self):
        json_output_1 = self.openstack('network flavor profile create ' + '--description ' + self.DESCRIPTION + ' ' + '--enable ' + '--metainfo ' + self.METAINFO, parse_output=True)
        ID = json_output_1.get('id')
        self.assertIsNotNone(ID)
        json_output = self.openstack('network flavor profile show ' + ID, parse_output=True)
        self.assertEqual(ID, json_output['id'])
        self.assertTrue(json_output['enabled'])
        self.assertEqual('fakedescription', json_output['description'])
        self.assertEqual('Extrainfo', json_output['meta_info'])
        raw_output = self.openstack('network flavor profile delete ' + ID)
        self.assertOutput('', raw_output)