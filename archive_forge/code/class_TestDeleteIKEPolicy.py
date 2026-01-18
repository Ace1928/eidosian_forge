from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ikepolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestDeleteIKEPolicy(TestIKEPolicy, common.TestDeleteVPNaaS):

    def setUp(self):
        super(TestDeleteIKEPolicy, self).setUp()
        self.networkclient.delete_vpn_ike_policy = mock.Mock()
        self.mocked = self.networkclient.delete_vpn_ike_policy
        self.cmd = ikepolicy.DeleteIKEPolicy(self.app, self.namespace)