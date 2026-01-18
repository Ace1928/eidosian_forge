import uuid
from openstackclient.tests.functional.network.v2 import common
class IPAvailabilityTests(common.NetworkTests):
    """Functional tests for IP availability"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if cls.haz_network:
            cls.NAME = uuid.uuid4().hex
            cls.NETWORK_NAME = uuid.uuid4().hex
            cls.openstack('network create ' + cls.NETWORK_NAME)
            cmd_output = cls.openstack('subnet create ' + '--network ' + cls.NETWORK_NAME + ' ' + '--subnet-range 10.10.10.0/24 ' + cls.NAME, parse_output=True)
            cls.assertOutput(cls.NAME, cmd_output['name'])

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.haz_network:
                raw_subnet = cls.openstack('subnet delete ' + cls.NAME)
                raw_network = cls.openstack('network delete ' + cls.NETWORK_NAME)
                cls.assertOutput('', raw_subnet)
                cls.assertOutput('', raw_network)
        finally:
            super().tearDownClass()

    def test_ip_availability_list(self):
        """Test ip availability list"""
        cmd_output = self.openstack('ip availability list', parse_output=True)
        names = [x['Network Name'] for x in cmd_output]
        self.assertIn(self.NETWORK_NAME, names)

    def test_ip_availability_show(self):
        """Test ip availability show"""
        cmd_output = self.openstack('ip availability show ' + self.NETWORK_NAME, parse_output=True)
        self.assertEqual(self.NETWORK_NAME, cmd_output['network_name'])