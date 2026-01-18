from unittest import mock
from osc_lib.cli import format_columns
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ipsec_site_connection
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestListIPsecSiteConn(TestIPsecSiteConn):

    def setUp(self):
        super(TestListIPsecSiteConn, self).setUp()
        self.cmd = ipsec_site_connection.ListIPsecSiteConnection(self.app, self.namespace)
        self.short_header = ('ID', 'Name', 'Peer Address', 'Authentication Algorithm', 'Status')
        self.short_data = (_ipsec_site_conn['id'], _ipsec_site_conn['name'], _ipsec_site_conn['peer_address'], _ipsec_site_conn['auth_mode'], _ipsec_site_conn['status'])
        self.networkclient.vpn_ipsec_site_connections = mock.Mock(return_value=[_ipsec_site_conn])
        self.mocked = self.networkclient.vpn_ipsec_site_connections

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