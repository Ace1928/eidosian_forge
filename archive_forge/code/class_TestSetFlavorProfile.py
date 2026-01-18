from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor_profile
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
class TestSetFlavorProfile(TestFlavorProfile):
    network_flavor_profile = network_fakes.create_one_service_profile()

    def setUp(self):
        super(TestSetFlavorProfile, self).setUp()
        self.network_client.update_service_profile = mock.Mock(return_value=None)
        self.network_client.find_service_profile = mock.Mock(return_value=self.network_flavor_profile)
        self.cmd = network_flavor_profile.SetNetworkFlavorProfile(self.app, self.namespace)

    def test_set_nothing(self):
        arglist = [self.network_flavor_profile.id]
        verifylist = [('flavor_profile', self.network_flavor_profile.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {}
        self.network_client.update_service_profile.assert_called_with(self.network_flavor_profile, **attrs)
        self.assertIsNone(result)

    def test_set_enable(self):
        arglist = ['--enable', self.network_flavor_profile.id]
        verifylist = [('enable', True), ('flavor_profile', self.network_flavor_profile.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'enabled': True}
        self.network_client.update_service_profile.assert_called_with(self.network_flavor_profile, **attrs)
        self.assertIsNone(result)

    def test_set_disable(self):
        arglist = ['--disable', self.network_flavor_profile.id]
        verifylist = [('disable', True), ('flavor_profile', self.network_flavor_profile.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'enabled': False}
        self.network_client.update_service_profile.assert_called_with(self.network_flavor_profile, **attrs)
        self.assertIsNone(result)