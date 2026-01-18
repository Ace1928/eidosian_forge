from unittest import mock
from openstackclient.compute.v2 import hypervisor_stats
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
class TestHypervisorStats(compute_fakes.TestComputev2):

    def setUp(self):
        super(TestHypervisorStats, self).setUp()
        self.compute_sdk_client.get = mock.Mock()