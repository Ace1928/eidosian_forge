from unittest import mock
from openstackclient.compute.v2 import hypervisor_stats
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
class TestHypervisorStatsShow(TestHypervisorStats):
    _stats = create_one_hypervisor_stats()

    def setUp(self):
        super(TestHypervisorStatsShow, self).setUp()
        self.compute_sdk_client.get.return_value = fakes.FakeResponse(data={'hypervisor_statistics': self._stats})
        self.cmd = hypervisor_stats.ShowHypervisorStats(self.app, None)
        self.columns = ('count', 'current_workload', 'disk_available_least', 'free_disk_gb', 'free_ram_mb', 'local_gb', 'local_gb_used', 'memory_mb', 'memory_mb_used', 'running_vms', 'vcpus', 'vcpus_used')
        self.data = (2, 0, 50, 100, 23000, 100, 0, 23800, 1400, 3, 8, 3)

    def test_hypervisor_show_stats(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)