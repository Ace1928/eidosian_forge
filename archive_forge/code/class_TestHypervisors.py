from novaclient.tests.functional import base
from novaclient import utils
class TestHypervisors(base.ClientTestBase):
    COMPUTE_API_VERSION = '2.1'

    def _test_list(self, cpu_info_type, uuid_as_id=False):
        hypervisors = self.client.hypervisors.list()
        if not len(hypervisors):
            self.fail('No hypervisors detected.')
        for hypervisor in hypervisors:
            self.assertIsInstance(hypervisor.cpu_info, cpu_info_type)
            if uuid_as_id:
                self.assertFalse(utils.is_integer_like(hypervisor.id), 'Expected hypervisor.id to be a UUID.')
                self.assertFalse(utils.is_integer_like(hypervisor.service['id']), 'Expected hypervisor.service.id to be a UUID.')
            else:
                self.assertTrue(utils.is_integer_like(hypervisor.id), 'Expected hypervisor.id to be an integer.')
                self.assertTrue(utils.is_integer_like(hypervisor.service['id']), 'Expected hypervisor.service.id to be an integer.')

    def test_list(self):
        self._test_list(str)