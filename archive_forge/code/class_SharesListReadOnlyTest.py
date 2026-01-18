import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.ddt
class SharesListReadOnlyTest(base.BaseTestCase):

    @ddt.data('admin', 'user')
    def test_shares_list(self, role):
        self.clients[role].manila('list')

    @ddt.data('admin', 'user')
    def test_list_with_debug_flag(self, role):
        self.clients[role].manila('list', flags='--debug')

    @ddt.data('admin', 'user')
    def test_shares_list_all_tenants(self, role):
        self.clients[role].manila('list', params='--all-tenants')

    @ddt.data('admin', 'user')
    def test_shares_list_filter_by_name(self, role):
        self.clients[role].manila('list', params='--name name')

    @ddt.data('admin', 'user')
    def test_shares_list_filter_by_export_location(self, role):
        self.clients[role].manila('list', params='--export_location fake')

    @ddt.data('admin', 'user')
    def test_shares_list_filter_by_inexact_name(self, role):
        self.clients[role].manila('list', params='--name~ na')

    @ddt.data('admin', 'user')
    def test_shares_list_filter_by_inexact_description(self, role):
        self.clients[role].manila('list', params='--description~ des')

    @ddt.data('admin', 'user')
    def test_shares_list_filter_by_status(self, role):
        self.clients[role].manila('list', params='--status status')

    def test_shares_list_filter_by_share_server_as_admin(self):
        self.clients['admin'].manila('list', params='--share-server fake')

    def test_shares_list_filter_by_share_server_as_user(self):
        self.assertRaises(exceptions.CommandFailed, self.clients['user'].manila, 'list', params='--share-server fake')

    @ddt.data('admin', 'user')
    def test_shares_list_filter_by_project_id(self, role):
        self.clients[role].manila('list', params='--project-id fake')

    def test_shares_list_filter_by_host(self):
        self.clients['admin'].manila('list', params='--host fake')

    @ddt.data('admin', 'user')
    def test_shares_list_with_limit_and_offset(self, role):
        self.clients[role].manila('list', params='--limit 1 --offset 1')

    @ddt.data({'role': 'admin', 'direction': 'asc'}, {'role': 'admin', 'direction': 'desc'}, {'role': 'user', 'direction': 'asc'}, {'role': 'user', 'direction': 'desc'})
    @ddt.unpack
    def test_shares_list_with_sorting(self, role, direction):
        self.clients[role].manila('list', params='--sort-key host --sort-dir ' + direction)

    @ddt.data('admin', 'user')
    def test_snapshot_list(self, role):
        self.clients[role].manila('snapshot-list')

    @ddt.data('admin', 'user')
    def test_snapshot_list_all_tenants(self, role):
        self.clients[role].manila('snapshot-list', params='--all-tenants')

    @ddt.data('admin', 'user')
    def test_snapshot_list_filter_by_name(self, role):
        self.clients[role].manila('snapshot-list', params='--name name')

    @ddt.data('admin', 'user')
    def test_snapshot_list_filter_by_status(self, role):
        self.clients[role].manila('snapshot-list', params='--status status')