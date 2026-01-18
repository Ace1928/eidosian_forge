import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
class TestServersAutoAllocateNetworkCLI(base.ClientTestBase):
    COMPUTE_API_VERSION = '2.37'

    def _find_network_in_table(self, table):
        for line in table.split('\n'):
            if '|' in line:
                l_property, l_value = line.split('|')[1:3]
                if ' network' in l_property.strip():
                    return ' '.join(l_property.strip().split()[:-1])

    def test_boot_server_with_auto_network(self):
        """Tests that the CLI defaults to 'auto' when --nic isn't specified.
        """
        if self.multiple_networks:
            self.skipTest('multiple networks available')
        server_info = self.nova('boot', params='%(name)s --flavor %(flavor)s --poll --image %(image)s ' % {'name': self.name_generate(), 'flavor': self.flavor.id, 'image': self.image.id})
        server_id = self._get_value_from_the_table(server_info, 'id')
        self.addCleanup(self.wait_for_resource_delete, server_id, self.client.servers)
        self.addCleanup(self.client.servers.delete, server_id)
        server_info = self.nova('show', params=server_id)
        network = self._find_network_in_table(server_info)
        self.assertIsNotNone(network, 'Auto-allocated network not found: %s' % server_info)

    def test_boot_server_with_no_network(self):
        """Tests that '--nic none' is honored.
        """
        server_info = self.nova('boot', params='%(name)s --flavor %(flavor)s --poll --image %(image)s --nic none' % {'name': self.name_generate(), 'flavor': self.flavor.id, 'image': self.image.id})
        server_id = self._get_value_from_the_table(server_info, 'id')
        self.addCleanup(self.wait_for_resource_delete, server_id, self.client.servers)
        self.addCleanup(self.client.servers.delete, server_id)
        server_info = self.nova('show', params=server_id)
        network = self._find_network_in_table(server_info)
        self.assertIsNone(network, 'Unexpected network allocation: %s' % server_info)