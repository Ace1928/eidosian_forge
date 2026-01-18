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
class ShareServersReadWriteBase(base.BaseTestCase):
    protocol = None

    def setUp(self):
        super(ShareServersReadWriteBase, self).setUp()
        if not CONF.run_share_servers_tests:
            message = 'share-server tests are disabled.'
            raise self.skipException(message)
        if self.protocol not in CONF.enable_protocols:
            message = '%s tests are disabled.' % self.protocol
            raise self.skipException(message)
        self.client = self.get_admin_client()
        if not self.client.share_network:
            message = 'Can run only with DHSS=True mode'
            raise self.skipException(message)

    def _create_share_and_share_network(self):
        name = data_utils.rand_name('autotest_share_name')
        description = data_utils.rand_name('autotest_share_description')
        common_share_network = self.client.get_share_network(self.client.share_network)
        share_net_info = utils.get_default_subnet(self.user_client, common_share_network['id']) if utils.share_network_subnets_are_supported() else common_share_network
        neutron_net_id = share_net_info['neutron_net_id'] if 'none' not in share_net_info['neutron_net_id'].lower() else None
        neutron_subnet_id = share_net_info['neutron_subnet_id'] if 'none' not in share_net_info['neutron_subnet_id'].lower() else None
        share_network = self.client.create_share_network(neutron_net_id=neutron_net_id, neutron_subnet_id=neutron_subnet_id)
        self.share = self.create_share(share_protocol=self.protocol, size=1, name=name, description=description, share_network=share_network['id'], client=self.client, wait_for_creation=True)
        self.share = self.client.get_share(self.share['id'])
        return (self.share, share_network)

    def _delete_share_and_share_server(self, share_id, share_server_id):
        self.client.delete_share(share_id)
        self.client.wait_for_share_deletion(share_id)
        self.client.delete_share_server(share_server_id)
        self.client.wait_for_share_server_deletion(share_server_id)

    def test_get_and_delete_share_server(self):
        self.share, share_network = self._create_share_and_share_network()
        share_server_id = self.client.get_share(self.share['id'])['share_server_id']
        server = self.client.get_share_server(share_server_id)
        expected_keys = ('id', 'host', 'status', 'created_at', 'updated_at', 'share_network_id', 'share_network_name', 'project_id')
        if utils.is_microversion_supported('2.49'):
            expected_keys += ('identifier', 'is_auto_deletable')
        for key in expected_keys:
            self.assertIn(key, server)
        self._delete_share_and_share_server(self.share['id'], share_server_id)
        self.client.delete_share_network(share_network['id'])

    @testtools.skipUnless(CONF.run_manage_tests, 'Share Manage/Unmanage tests are disabled.')
    @utils.skip_if_microversion_not_supported('2.49')
    def test_manage_and_unmanage_share_server(self):
        share, share_network = self._create_share_and_share_network()
        share_server_id = self.client.get_share(self.share['id'])['share_server_id']
        server = self.client.get_share_server(share_server_id)
        server_host = server['host']
        export_location = self.client.list_share_export_locations(self.share['id'])[0]['Path']
        share_host = share['host']
        identifier = server['identifier']
        self.assertEqual('True', server['is_auto_deletable'])
        self.client.unmanage_share(share['id'])
        self.client.wait_for_share_deletion(share['id'])
        server = self.client.get_share_server(share_server_id)
        self.assertEqual('False', server['is_auto_deletable'])
        self.client.unmanage_server(share_server_id)
        self.client.wait_for_share_server_deletion(share_server_id)
        managed_share_server_id = self.client.share_server_manage(server_host, share_network['id'], identifier)
        self.client.wait_for_resource_status(managed_share_server_id, constants.STATUS_ACTIVE, resource_type='share_server')
        managed_server = self.client.get_share_server(managed_share_server_id)
        self.assertEqual('False', managed_server['is_auto_deletable'])
        managed_share_id = self.client.manage_share(share_host, self.protocol, export_location, managed_share_server_id)
        self.client.wait_for_resource_status(managed_share_id, constants.STATUS_AVAILABLE)
        self._delete_share_and_share_server(managed_share_id, managed_share_server_id)
        self.client.delete_share_network(share_network['id'])