import random
import uuid
from openstackclient.tests.functional.network.v2 import common
class FloatingIpTests(common.NetworkTests):
    """Functional tests for floating ip"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if cls.haz_network:
            cls.EXTERNAL_NETWORK_NAME = uuid.uuid4().hex
            cls.PRIVATE_NETWORK_NAME = uuid.uuid4().hex
            json_output = cls.openstack('network create ' + '--external ' + cls.EXTERNAL_NETWORK_NAME, parse_output=True)
            cls.external_network_id = json_output['id']
            json_output = cls.openstack('network create ' + cls.PRIVATE_NETWORK_NAME, parse_output=True)
            cls.private_network_id = json_output['id']

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.haz_network:
                del_output = cls.openstack('network delete ' + cls.EXTERNAL_NETWORK_NAME + ' ' + cls.PRIVATE_NETWORK_NAME)
                cls.assertOutput('', del_output)
        finally:
            super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.assertIsNotNone(self.external_network_id)
        self.assertIsNotNone(self.private_network_id)

    def _create_subnet(self, network_name, subnet_name):
        subnet_id = None
        for i in range(4):
            subnet = '.'.join(map(str, (random.randint(0, 223) for _ in range(3)))) + '.0/26'
            try:
                json_output = self.openstack('subnet create ' + '--network ' + network_name + ' ' + '--subnet-range ' + subnet + ' ' + subnet_name, parse_output=True)
                self.assertIsNotNone(json_output['id'])
                subnet_id = json_output['id']
            except Exception:
                if i == 3:
                    raise
                pass
            else:
                break
        return subnet_id

    def test_floating_ip_delete(self):
        """Test create, delete multiple"""
        ext_subnet_id = self._create_subnet(self.EXTERNAL_NETWORK_NAME, 'ext-test-delete')
        self.addCleanup(self.openstack, 'subnet delete ' + ext_subnet_id)
        json_output = self.openstack('floating ip create ' + '--description aaaa ' + self.EXTERNAL_NETWORK_NAME, parse_output=True)
        self.assertIsNotNone(json_output['id'])
        ip1 = json_output['id']
        self.assertEqual('aaaa', json_output['description'])
        json_output = self.openstack('floating ip create ' + '--description bbbb ' + self.EXTERNAL_NETWORK_NAME, parse_output=True)
        self.assertIsNotNone(json_output['id'])
        ip2 = json_output['id']
        self.assertEqual('bbbb', json_output['description'])
        del_output = self.openstack('floating ip delete ' + ip1 + ' ' + ip2)
        self.assertOutput('', del_output)
        self.assertIsNotNone(json_output['floating_network_id'])

    def test_floating_ip_list(self):
        """Test create defaults, list filters, delete"""
        ext_subnet_id = self._create_subnet(self.EXTERNAL_NETWORK_NAME, 'ext-test-delete')
        self.addCleanup(self.openstack, 'subnet delete ' + ext_subnet_id)
        json_output = self.openstack('floating ip create ' + '--description aaaa ' + self.EXTERNAL_NETWORK_NAME, parse_output=True)
        self.assertIsNotNone(json_output['id'])
        ip1 = json_output['id']
        self.addCleanup(self.openstack, 'floating ip delete ' + ip1)
        self.assertEqual('aaaa', json_output['description'])
        self.assertIsNotNone(json_output['floating_network_id'])
        fip1 = json_output['floating_ip_address']
        json_output = self.openstack('floating ip create ' + '--description bbbb ' + self.EXTERNAL_NETWORK_NAME, parse_output=True)
        self.assertIsNotNone(json_output['id'])
        ip2 = json_output['id']
        self.addCleanup(self.openstack, 'floating ip delete ' + ip2)
        self.assertEqual('bbbb', json_output['description'])
        self.assertIsNotNone(json_output['floating_network_id'])
        fip2 = json_output['floating_ip_address']
        json_output = self.openstack('floating ip list', parse_output=True)
        fip_map = {item.get('ID'): item.get('Floating IP Address') for item in json_output}
        self.assertIn(ip1, fip_map.keys())
        self.assertIn(ip2, fip_map.keys())
        self.assertIn(fip1, fip_map.values())
        self.assertIn(fip2, fip_map.values())
        json_output = self.openstack('floating ip list ' + '--long', parse_output=True)
        fip_map = {item.get('ID'): item.get('Floating IP Address') for item in json_output}
        self.assertIn(ip1, fip_map.keys())
        self.assertIn(ip2, fip_map.keys())
        self.assertIn(fip1, fip_map.values())
        self.assertIn(fip2, fip_map.values())
        desc_map = {item.get('ID'): item.get('Description') for item in json_output}
        self.assertIn('aaaa', desc_map.values())
        self.assertIn('bbbb', desc_map.values())
        json_output = self.openstack('floating ip show ' + ip1, parse_output=True)
        self.assertIsNotNone(json_output['id'])
        self.assertEqual(ip1, json_output['id'])
        self.assertEqual('aaaa', json_output['description'])
        self.assertIsNotNone(json_output['floating_network_id'])
        self.assertEqual(fip1, json_output['floating_ip_address'])

    def test_floating_ip_set_and_unset_port(self):
        """Test Floating IP Set and Unset port"""
        ext_subnet_id = self._create_subnet(self.EXTERNAL_NETWORK_NAME, 'ext-test-delete')
        self.addCleanup(self.openstack, 'subnet delete ' + ext_subnet_id)
        priv_subnet_id = self._create_subnet(self.PRIVATE_NETWORK_NAME, 'priv-test-delete')
        self.addCleanup(self.openstack, 'subnet delete ' + priv_subnet_id)
        self.ROUTER = uuid.uuid4().hex
        self.PORT_NAME = uuid.uuid4().hex
        json_output = self.openstack('floating ip create ' + '--description aaaa ' + self.EXTERNAL_NETWORK_NAME, parse_output=True)
        self.assertIsNotNone(json_output['id'])
        ip1 = json_output['id']
        self.addCleanup(self.openstack, 'floating ip delete ' + ip1)
        self.assertEqual('aaaa', json_output['description'])
        json_output = self.openstack('port create ' + '--network ' + self.PRIVATE_NETWORK_NAME + ' ' + '--fixed-ip subnet=' + priv_subnet_id + ' ' + self.PORT_NAME, parse_output=True)
        self.assertIsNotNone(json_output['id'])
        port_id = json_output['id']
        json_output = self.openstack('router create ' + self.ROUTER, parse_output=True)
        self.assertIsNotNone(json_output['id'])
        self.addCleanup(self.openstack, 'router delete ' + self.ROUTER)
        self.openstack('router add port ' + self.ROUTER + ' ' + port_id)
        self.openstack('router set ' + '--external-gateway ' + self.EXTERNAL_NETWORK_NAME + ' ' + self.ROUTER)
        self.addCleanup(self.openstack, 'router unset --external-gateway ' + self.ROUTER)
        self.addCleanup(self.openstack, 'router remove port ' + self.ROUTER + ' ' + port_id)
        self.openstack('floating ip set ' + '--port ' + port_id + ' ' + ip1)
        self.addCleanup(self.openstack, 'floating ip unset --port ' + ip1)
        json_output = self.openstack('floating ip show ' + ip1, parse_output=True)
        self.assertEqual(port_id, json_output['port_id'])