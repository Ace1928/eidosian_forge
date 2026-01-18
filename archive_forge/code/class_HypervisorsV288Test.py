from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import hypervisors as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
class HypervisorsV288Test(HypervisorsV253Test):
    data_fixture_class = data.V288

    def setUp(self):
        super().setUp()
        self.cs.api_version = api_versions.APIVersion('2.88')

    def test_hypervisor_uptime(self):
        expected = {'id': self.data_fixture.hyper_id_1, 'hypervisor_hostname': 'hyper1', 'uptime': 'fake uptime', 'state': 'up', 'status': 'enabled'}
        result = self.cs.hypervisors.uptime(self.data_fixture.hyper_id_1)
        self.assert_request_id(result, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('GET', '/os-hypervisors/%s' % self.data_fixture.hyper_id_1)
        self.compare_to_expected(expected, result)

    def test_hypervisor_statistics(self):
        exc = self.assertRaises(exceptions.UnsupportedVersion, self.cs.hypervisor_stats.statistics)
        self.assertIn("The 'statistics' API is removed in API version 2.88 or later.", str(exc))