from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import l3_conntrack_helper
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestConntrackHelper(network_fakes.TestNetworkV2):

    def setUp(self):
        super(TestConntrackHelper, self).setUp()
        self.router = network_fakes.FakeRouter.create_one_router()
        self.network_client.find_router = mock.Mock(return_value=self.router)