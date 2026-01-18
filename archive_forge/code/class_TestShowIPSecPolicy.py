from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ipsecpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestShowIPSecPolicy(TestIPSecPolicy, common.TestShowVPNaaS):

    def setUp(self):
        super(TestShowIPSecPolicy, self).setUp()
        self.networkclient.get_vpn_ipsec_policy = mock.Mock(return_value=_ipsecpolicy)
        self.mocked = self.networkclient.get_vpn_ipsec_policy
        self.cmd = ipsecpolicy.ShowIPsecPolicy(self.app, self.namespace)