from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ikepolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestListIKEPolicy(TestIKEPolicy):

    def setUp(self):
        super(TestListIKEPolicy, self).setUp()
        self.cmd = ikepolicy.ListIKEPolicy(self.app, self.namespace)
        self.short_header = ('ID', 'Name', 'Authentication Algorithm', 'Encryption Algorithm', 'IKE Version', 'Perfect Forward Secrecy (PFS)')
        self.short_data = (_ikepolicy['id'], _ikepolicy['name'], _ikepolicy['auth_algorithm'], _ikepolicy['encryption_algorithm'], _ikepolicy['ike_version'], _ikepolicy['pfs'])
        self.networkclient.vpn_ike_policies = mock.Mock(return_value=[_ikepolicy])
        self.mocked = self.networkclient.vpn_ike_policies

    def test_list_with_long_option(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.headers), headers)

    def test_list_with_no_option(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.short_header), headers)
        self.assertEqual([self.short_data], list(data))