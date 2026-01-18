from unittest import mock
import uuid
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import vpnservice
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestSetVPNService(TestVPNService, common.TestSetVPNaaS):

    def setUp(self):
        super(TestSetVPNService, self).setUp()
        self.networkclient.update_vpn_service = mock.Mock(return_value=_vpnservice)
        self.mocked = self.networkclient.update_vpn_service
        self.cmd = vpnservice.SetVPNSercice(self.app, self.namespace)

    def test_set_name(self):
        target = self.resource['id']
        update = 'change'
        arglist = [target, '--name', update]
        verifylist = [(self.res, target), ('name', update)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'name': update})
        self.assertIsNone(result)