from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestRemoveNetworkFlavorFromProfile(TestNetworkFlavor):
    network_flavor = network_fakes.create_one_network_flavor()
    service_profile = network_fakes.create_one_service_profile()

    def setUp(self):
        super(TestRemoveNetworkFlavorFromProfile, self).setUp()
        self.network_client.find_flavor = mock.Mock(return_value=self.network_flavor)
        self.network_client.find_service_profile = mock.Mock(return_value=self.service_profile)
        self.network_client.disassociate_flavor_from_service_profile = mock.Mock()
        self.cmd = network_flavor.RemoveNetworkFlavorFromProfile(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_remove_flavor_from_service_profile(self):
        arglist = [self.network_flavor.id, self.service_profile.id]
        verifylist = [('flavor', self.network_flavor.id), ('service_profile', self.service_profile.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.network_client.disassociate_flavor_from_service_profile.assert_called_once_with(self.network_flavor, self.service_profile)