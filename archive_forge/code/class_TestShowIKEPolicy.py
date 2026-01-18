from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ikepolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestShowIKEPolicy(TestIKEPolicy, common.TestShowVPNaaS):

    def setUp(self):
        super(TestShowIKEPolicy, self).setUp()
        self.networkclient.get_vpn_ike_policy = mock.Mock(return_value=_ikepolicy)
        self.mocked = self.networkclient.get_vpn_ike_policy
        self.cmd = ikepolicy.ShowIKEPolicy(self.app, self.namespace)