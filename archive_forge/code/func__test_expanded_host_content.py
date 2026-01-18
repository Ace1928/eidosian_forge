from openstack.cloud import inventory
from openstack.tests.functional import base
def _test_expanded_host_content(self, host):
    self.assertEqual(host['image']['name'], self.image.name)
    self.assertEqual(host['flavor']['name'], self.flavor.name)