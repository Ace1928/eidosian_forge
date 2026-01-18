import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.ddt
class SharesTestMigration(base.BaseTestCase):

    def setUp(self):
        super(SharesTestMigration, self).setUp()
        self.old_type = self.create_share_type(data_utils.rand_name('test_share_type'), driver_handles_share_servers=True)
        self.new_type = self.create_share_type(data_utils.rand_name('test_share_type'), driver_handles_share_servers=True)
        self.error_type = self.create_share_type(data_utils.rand_name('test_share_type'), driver_handles_share_servers=True, extra_specs={'cause_error': 'no_valid_host'})
        self.old_share_net = self.get_user_client().get_share_network(self.get_user_client().share_network)
        share_net_info = utils.get_default_subnet(self.get_user_client(), self.old_share_net['id']) if utils.share_network_subnets_are_supported() else self.old_share_net
        self.new_share_net = self.create_share_network(neutron_net_id=share_net_info['neutron_net_id'], neutron_subnet_id=share_net_info['neutron_subnet_id'])

    @utils.skip_if_microversion_not_supported('2.22')
    @ddt.data('migration_error', 'migration_success', 'None')
    def test_reset_task_state(self, state):
        share = self.create_share(share_protocol='nfs', size=1, name=data_utils.rand_name('autotest_share_name'), client=self.get_user_client(), share_type=self.old_type['ID'], share_network=self.old_share_net['id'], wait_for_creation=True)
        share = self.user_client.get_share(share['id'])
        self.admin_client.reset_task_state(share['id'], state)
        share = self.user_client.get_share(share['id'])
        self.assertEqual(state, share['task_state'])

    @utils.skip_if_microversion_not_supported('2.29')
    @testtools.skipUnless(CONF.run_migration_tests, 'Share migration tests are disabled.')
    @ddt.data('cancel', 'success', 'error')
    def test_full_migration(self, test_type):
        share = self.create_share(share_protocol='nfs', size=1, name=data_utils.rand_name('autotest_share_name'), client=self.get_user_client(), share_type=self.old_type['ID'], share_network=self.old_share_net['id'], wait_for_creation=True)
        share = self.admin_client.get_share(share['id'])
        pools = self.admin_client.pool_list(detail=True)
        dest_pool = utils.choose_matching_backend(share, pools, self.new_type)
        self.assertIsNotNone(dest_pool)
        source_pool = share['host']
        new_type = self.new_type
        if test_type == 'error':
            statuses = constants.TASK_STATE_MIGRATION_ERROR
            new_type = self.error_type
        else:
            statuses = (constants.TASK_STATE_MIGRATION_DRIVER_PHASE1_DONE, constants.TASK_STATE_DATA_COPYING_COMPLETED)
        self.admin_client.migration_start(share['id'], dest_pool, writable=True, nondisruptive=False, preserve_metadata=True, preserve_snapshots=True, force_host_assisted_migration=False, new_share_network=self.new_share_net['id'], new_share_type=new_type['ID'])
        share = self.admin_client.wait_for_migration_task_state(share['id'], dest_pool, statuses)
        progress = self.admin_client.migration_get_progress(share['id'])
        self.assertEqual('100', progress['total_progress'])
        self.assertEqual(source_pool, share['host'])
        self.assertEqual(self.old_type['ID'], share['share_type'])
        self.assertEqual(self.old_share_net['id'], share['share_network_id'])
        if test_type == 'error':
            self.assertEqual(statuses, progress['task_state'])
        else:
            if test_type == 'success':
                self.admin_client.migration_complete(share['id'])
                statuses = constants.TASK_STATE_MIGRATION_SUCCESS
            elif test_type == 'cancel':
                self.admin_client.migration_cancel(share['id'])
                statuses = constants.TASK_STATE_MIGRATION_CANCELLED
            share = self.admin_client.wait_for_migration_task_state(share['id'], dest_pool, statuses)
            progress = self.admin_client.migration_get_progress(share['id'])
            self.assertEqual(statuses, progress['task_state'])
            if test_type == 'success':
                self.assertEqual(dest_pool, share['host'])
                self.assertEqual(new_type['ID'], share['share_type'])
                self.assertEqual(self.new_share_net['id'], share['share_network_id'])
            else:
                self.assertEqual(source_pool, share['host'])
                self.assertEqual(self.old_type['ID'], share['share_type'])
                self.assertEqual(self.old_share_net['id'], share['share_network_id'])