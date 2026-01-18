from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ipsecpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestDeleteIPSecPolicy(TestIPSecPolicy, common.TestDeleteVPNaaS):

    def setUp(self):
        super(TestDeleteIPSecPolicy, self).setUp()
        self.networkclient.delete_vpn_ipsec_policy = mock.Mock()
        self.mocked = self.networkclient.delete_vpn_ipsec_policy
        self.cmd = ipsecpolicy.DeleteIPsecPolicy(self.app, self.namespace)