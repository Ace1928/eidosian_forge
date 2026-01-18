import ast
import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.ddt
class ShareServersReadOnlyTest(base.BaseTestCase):

    def setUp(self):
        super(ShareServersReadOnlyTest, self).setUp()
        self.client = self.get_admin_client()

    def test_share_server_list(self):
        self.client.list_share_servers()

    def test_share_server_list_with_host_param(self):
        self.client.list_share_servers(filters={'host': 'fake_host'})

    def test_share_server_list_with_status_param(self):
        self.client.list_share_servers(filters={'status': 'fake_status'})

    def test_share_server_list_with_share_network_param(self):
        self.client.list_share_servers(filters={'share_network': 'fake_sn'})

    def test_share_server_list_with_project_id_param(self):
        self.client.list_share_servers(filters={'project_id': 'fake_project_id'})

    @ddt.data('host', 'status', 'project_id', 'share_network', 'host,status,project_id,share_network')
    def test_share_server_list_with_specified_columns(self, columns):
        self.client.list_share_servers(columns=columns)

    def test_share_server_list_by_user(self):
        self.assertRaises(exceptions.CommandFailed, self.user_client.list_share_servers)