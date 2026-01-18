from unittest import mock
from osc_lib.cli import format_columns
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import ipsec_site_connection
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestShowIPsecSiteConn(TestIPsecSiteConn, common.TestShowVPNaaS):

    def setUp(self):
        super(TestShowIPsecSiteConn, self).setUp()
        self.networkclient.get_vpn_ipsec_site_connection = mock.Mock(return_value=_ipsec_site_conn)
        self.mocked = self.networkclient.get_vpn_ipsec_site_connection
        self.cmd = ipsec_site_connection.ShowIPsecSiteConnection(self.app, self.namespace)