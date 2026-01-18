from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.tests.functional import base
class ManilaClientTestSchedulerStatsReadOnly(base.BaseTestCase):

    def test_pools_list(self):
        self.clients['admin'].manila('pool-list')

    def test_pools_list_with_debug_flag(self):
        self.clients['admin'].manila('pool-list', flags='--debug')

    def test_pools_list_with_detail(self):
        self.clients['admin'].manila('pool-list', params='--detail')

    def test_pools_list_with_share_type_filter(self):
        share_type = self.create_share_type(name=data_utils.rand_name('manilaclient_functional_test'), snapshot_support=True)
        self.clients['admin'].manila('pool-list', params='--share_type ' + share_type['ID'])

    def test_pools_list_with_filters(self):
        self.clients['admin'].manila('pool-list', params='--host myhost --backend mybackend --pool mypool')

    def test_pools_list_by_user(self):
        self.assertRaises(exceptions.CommandFailed, self.clients['user'].manila, 'pool-list')