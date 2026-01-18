from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor_profile
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
class TestShowFlavorProfile(TestFlavorProfile):
    network_flavor_profile = network_fakes.create_one_service_profile()
    columns = ('description', 'driver', 'enabled', 'id', 'meta_info', 'project_id')
    data = (network_flavor_profile.description, network_flavor_profile.driver, network_flavor_profile.is_enabled, network_flavor_profile.id, network_flavor_profile.meta_info, network_flavor_profile.project_id)

    def setUp(self):
        super(TestShowFlavorProfile, self).setUp()
        self.network_client.find_service_profile = mock.Mock(return_value=self.network_flavor_profile)
        self.cmd = network_flavor_profile.ShowNetworkFlavorProfile(self.app, self.namespace)

    def test_show_all_options(self):
        arglist = [self.network_flavor_profile.id]
        verifylist = [('flavor_profile', self.network_flavor_profile.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_service_profile.assert_called_once_with(self.network_flavor_profile.id, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)