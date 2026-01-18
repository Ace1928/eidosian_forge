from unittest import mock
from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.engine.clients.os import neutron
from heat.engine.clients.os.neutron import lbaas_constraints as lc
from heat.engine.clients.os.neutron import neutron_constraints as nc
from heat.tests import common
from heat.tests import utils
class NeutronClientPluginTest(NeutronClientPluginTestCase):

    def setUp(self):
        super(NeutronClientPluginTest, self).setUp()
        self.mock_find = self.patchobject(neutron.neutronV20, 'find_resourceid_by_name_or_id')
        self.mock_find.return_value = 42

    def test_get_secgroup_uuids(self):
        sgs_uuid = ['b62c3079-6946-44f5-a67b-6b9091884d4f', '9887157c-d092-40f5-b547-6361915fce7d']
        sgs_list = self.neutron_plugin.get_secgroup_uuids(sgs_uuid)
        self.assertEqual(sgs_uuid, sgs_list)
        sgs_non_uuid = ['security_group_1']
        expected_groups = ['0389f747-7785-4757-b7bb-2ab07e4b09c3']
        fake_list = {'security_groups': [{'tenant_id': 'test_tenant_id', 'id': '0389f747-7785-4757-b7bb-2ab07e4b09c3', 'name': 'security_group_1', 'security_group_rules': [], 'description': 'no protocol'}]}
        self.neutron_client.list_security_groups.return_value = fake_list
        self.assertEqual(expected_groups, self.neutron_plugin.get_secgroup_uuids(sgs_non_uuid))
        fake_list = {'security_groups': [{'tenant_id': 'test_tenant_id', 'id': '0389f747-7785-4757-b7bb-2ab07e4b09c3', 'name': 'security_group_1', 'security_group_rules': [], 'description': 'no protocol'}, {'tenant_id': 'test_tenant_id', 'id': '384ccd91-447c-4d83-832c-06974a7d3d05', 'name': 'security_group_1', 'security_group_rules': [], 'description': 'no protocol'}]}
        self.neutron_client.list_security_groups.return_value = fake_list
        self.assertRaises(exception.PhysicalResourceNameAmbiguity, self.neutron_plugin.get_secgroup_uuids, sgs_non_uuid)

    def test_check_lb_status(self):
        self.neutron_client.show_loadbalancer.side_effect = [{'loadbalancer': {'provisioning_status': 'ACTIVE'}}, {'loadbalancer': {'provisioning_status': 'PENDING_CREATE'}}, {'loadbalancer': {'provisioning_status': 'ERROR'}}]
        self.assertTrue(self.neutron_plugin.check_lb_status('1234'))
        self.assertFalse(self.neutron_plugin.check_lb_status('1234'))
        self.assertRaises(exception.ResourceInError, self.neutron_plugin.check_lb_status, '1234')