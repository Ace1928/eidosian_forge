import json
from openstackclient.tests.functional import base
class HypervisorTests(base.TestCase):
    """Functional tests for hypervisor."""

    def test_hypervisor_list(self):
        """Test create defaults, list filters, delete"""
        cmd_output = json.loads(self.openstack('hypervisor list -f json --os-compute-api-version 2.1'))
        ids1 = [x['ID'] for x in cmd_output]
        self.assertIsNotNone(cmd_output)
        cmd_output = json.loads(self.openstack('hypervisor list -f json'))
        ids2 = [x['ID'] for x in cmd_output]
        self.assertIsNotNone(cmd_output)
        for i in ids1:
            cmd_output = json.loads(self.openstack('hypervisor show %s -f json  --os-compute-api-version 2.1' % i))
            self.assertIsNotNone(cmd_output)
        for i in ids2:
            cmd_output = json.loads(self.openstack('hypervisor show %s -f json' % i))
            self.assertIsNotNone(cmd_output)