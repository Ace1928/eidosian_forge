import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.common import quota
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
class TestQuotaList(TestQuota):
    """Test cases for quota list command"""
    compute_column_header = ('Project ID', 'Cores', 'Fixed IPs', 'Injected Files', 'Injected File Content Bytes', 'Injected File Path Bytes', 'Instances', 'Key Pairs', 'Metadata Items', 'Ram', 'Server Groups', 'Server Group Members')
    network_column_header = ('Project ID', 'Floating IPs', 'Networks', 'Ports', 'RBAC Policies', 'Routers', 'Security Groups', 'Security Group Rules', 'Subnets', 'Subnet Pools')
    volume_column_header = ('Project ID', 'Backups', 'Backup Gigabytes', 'Gigabytes', 'Per Volume Gigabytes', 'Snapshots', 'Volumes')

    def setUp(self):
        super(TestQuotaList, self).setUp()
        self.projects_mock.get.side_effect = self.projects
        self.projects_mock.list.return_value = self.projects
        self.compute_quotas = [compute_fakes.create_one_comp_quota(), compute_fakes.create_one_comp_quota()]
        self.compute_default_quotas = [compute_fakes.create_one_default_comp_quota(), compute_fakes.create_one_default_comp_quota()]
        self.compute_client.quotas.defaults = mock.Mock(side_effect=self.compute_default_quotas)
        self.compute_reference_data = (self.projects[0].id, self.compute_quotas[0].cores, self.compute_quotas[0].fixed_ips, self.compute_quotas[0].injected_files, self.compute_quotas[0].injected_file_content_bytes, self.compute_quotas[0].injected_file_path_bytes, self.compute_quotas[0].instances, self.compute_quotas[0].key_pairs, self.compute_quotas[0].metadata_items, self.compute_quotas[0].ram, self.compute_quotas[0].server_groups, self.compute_quotas[0].server_group_members)
        self.network_quotas = [network_fakes.FakeQuota.create_one_net_quota(), network_fakes.FakeQuota.create_one_net_quota()]
        self.network_default_quotas = [network_fakes.FakeQuota.create_one_default_net_quota(), network_fakes.FakeQuota.create_one_default_net_quota()]
        self.network_client.get_quota_default = mock.Mock(side_effect=self.network_default_quotas)
        self.network_reference_data = (self.projects[0].id, self.network_quotas[0].floating_ips, self.network_quotas[0].networks, self.network_quotas[0].ports, self.network_quotas[0].rbac_policies, self.network_quotas[0].routers, self.network_quotas[0].security_groups, self.network_quotas[0].security_group_rules, self.network_quotas[0].subnets, self.network_quotas[0].subnet_pools)
        self.volume_quotas = [volume_fakes.create_one_vol_quota(), volume_fakes.create_one_vol_quota()]
        self.volume_default_quotas = [volume_fakes.create_one_default_vol_quota(), volume_fakes.create_one_default_vol_quota()]
        self.volume_client.quotas.defaults = mock.Mock(side_effect=self.volume_default_quotas)
        self.volume_reference_data = (self.projects[0].id, self.volume_quotas[0].backups, self.volume_quotas[0].backup_gigabytes, self.volume_quotas[0].gigabytes, self.volume_quotas[0].per_volume_gigabytes, self.volume_quotas[0].snapshots, self.volume_quotas[0].volumes)
        self.cmd = quota.ListQuota(self.app, None)

    @staticmethod
    def _get_detailed_reference_data(quota):
        reference_data = []
        for name, values in quota.to_dict().items():
            if type(values) is dict:
                if 'used' in values:
                    in_use = values['used']
                else:
                    in_use = values['in_use']
                resource_values = [in_use, values['reserved'], values['limit']]
                reference_data.append(tuple([name] + resource_values))
        return reference_data

    def test_quota_list_details_compute(self):
        detailed_quota = compute_fakes.create_one_comp_detailed_quota()
        detailed_column_header = ('Resource', 'In Use', 'Reserved', 'Limit')
        detailed_reference_data = self._get_detailed_reference_data(detailed_quota)
        self.compute_client.quotas.get = mock.Mock(return_value=detailed_quota)
        arglist = ['--detail', '--compute']
        verifylist = [('detail', True), ('compute', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(detailed_column_header, columns)
        self.assertEqual(sorted(detailed_reference_data), sorted(ret_quotas))

    def test_quota_list_details_network(self):
        detailed_quota = network_fakes.FakeQuota.create_one_net_detailed_quota()
        detailed_column_header = ('Resource', 'In Use', 'Reserved', 'Limit')
        detailed_reference_data = self._get_detailed_reference_data(detailed_quota)
        self.network_client.get_quota = mock.Mock(return_value=detailed_quota)
        arglist = ['--detail', '--network']
        verifylist = [('detail', True), ('network', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(detailed_column_header, columns)
        self.assertEqual(sorted(detailed_reference_data), sorted(ret_quotas))

    def test_quota_list_details_volume(self):
        detailed_quota = volume_fakes.create_one_detailed_quota()
        detailed_column_header = ('Resource', 'In Use', 'Reserved', 'Limit')
        detailed_reference_data = self._get_detailed_reference_data(detailed_quota)
        self.volume_client.quotas.get = mock.Mock(return_value=detailed_quota)
        arglist = ['--detail', '--volume']
        verifylist = [('detail', True), ('volume', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(detailed_column_header, columns)
        self.assertEqual(sorted(detailed_reference_data), sorted(ret_quotas))

    def test_quota_list_compute(self):
        self.compute_client.quotas.get = mock.Mock(side_effect=self.compute_quotas)
        arglist = ['--compute']
        verifylist = [('compute', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.compute_column_header, columns)
        self.assertEqual(self.compute_reference_data, ret_quotas[0])
        self.assertEqual(2, len(ret_quotas))

    def test_quota_list_compute_default(self):
        self.compute_client.quotas.get = mock.Mock(side_effect=[self.compute_quotas[0], compute_fakes.create_one_default_comp_quota()])
        arglist = ['--compute']
        verifylist = [('compute', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.compute_column_header, columns)
        self.assertEqual(self.compute_reference_data, ret_quotas[0])
        self.assertEqual(1, len(ret_quotas))

    def test_quota_list_compute_no_project_not_found(self):
        self.compute_client.quotas.get = mock.Mock(side_effect=[self.compute_quotas[0], exceptions.NotFound('NotFound')])
        arglist = ['--compute']
        verifylist = [('compute', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.compute_column_header, columns)
        self.assertEqual(self.compute_reference_data, ret_quotas[0])
        self.assertEqual(1, len(ret_quotas))

    def test_quota_list_compute_no_project_4xx(self):
        self.compute_client.quotas.get = mock.Mock(side_effect=[self.compute_quotas[0], exceptions.BadRequest('Bad request')])
        arglist = ['--compute']
        verifylist = [('compute', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.compute_column_header, columns)
        self.assertEqual(self.compute_reference_data, ret_quotas[0])
        self.assertEqual(1, len(ret_quotas))

    def test_quota_list_compute_no_project_5xx(self):
        self.compute_client.quotas.get = mock.Mock(side_effect=[self.compute_quotas[0], exceptions.HTTPNotImplemented('Not implemented??')])
        arglist = ['--compute']
        verifylist = [('compute', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.HTTPNotImplemented, self.cmd.take_action, parsed_args)

    def test_quota_list_compute_by_project(self):
        self.compute_client.quotas.get = mock.Mock(side_effect=self.compute_quotas)
        arglist = ['--compute', '--project', self.projects[0].name]
        verifylist = [('compute', True), ('project', self.projects[0].name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.compute_column_header, columns)
        self.assertEqual(self.compute_reference_data, ret_quotas[0])
        self.assertEqual(1, len(ret_quotas))

    def test_quota_list_network(self):
        self.network_client.get_quota = mock.Mock(side_effect=self.network_quotas)
        arglist = ['--network']
        verifylist = [('network', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.network_column_header, columns)
        self.assertEqual(self.network_reference_data, ret_quotas[0])
        self.assertEqual(2, len(ret_quotas))

    def test_quota_list_network_default(self):
        self.network_client.get_quota = mock.Mock(side_effect=[self.network_quotas[0], network_fakes.FakeQuota.create_one_default_net_quota()])
        arglist = ['--network']
        verifylist = [('network', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.network_column_header, columns)
        self.assertEqual(self.network_reference_data, ret_quotas[0])
        self.assertEqual(1, len(ret_quotas))

    def test_quota_list_network_no_project(self):
        self.network_client.get_quota = mock.Mock(side_effect=[self.network_quotas[0], exceptions.NotFound('NotFound')])
        arglist = ['--network']
        verifylist = [('network', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.network_column_header, columns)
        self.assertEqual(self.network_reference_data, ret_quotas[0])
        self.assertEqual(1, len(ret_quotas))

    def test_quota_list_network_by_project(self):
        self.network_client.get_quota = mock.Mock(side_effect=self.network_quotas)
        arglist = ['--network', '--project', self.projects[0].name]
        verifylist = [('network', True), ('project', self.projects[0].name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.network_column_header, columns)
        self.assertEqual(self.network_reference_data, ret_quotas[0])
        self.assertEqual(1, len(ret_quotas))

    def test_quota_list_volume(self):
        self.volume_client.quotas.get = mock.Mock(side_effect=self.volume_quotas)
        arglist = ['--volume']
        verifylist = [('volume', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.volume_column_header, columns)
        self.assertEqual(self.volume_reference_data, ret_quotas[0])
        self.assertEqual(2, len(ret_quotas))

    def test_quota_list_volume_default(self):
        self.volume_client.quotas.get = mock.Mock(side_effect=[self.volume_quotas[0], volume_fakes.create_one_default_vol_quota()])
        arglist = ['--volume']
        verifylist = [('volume', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.volume_column_header, columns)
        self.assertEqual(self.volume_reference_data, ret_quotas[0])
        self.assertEqual(1, len(ret_quotas))

    def test_quota_list_volume_no_project(self):
        self.volume_client.quotas.get = mock.Mock(side_effect=[self.volume_quotas[0], volume_fakes.create_one_default_vol_quota()])
        arglist = ['--volume']
        verifylist = [('volume', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.volume_column_header, columns)
        self.assertEqual(self.volume_reference_data, ret_quotas[0])
        self.assertEqual(1, len(ret_quotas))

    def test_quota_list_volume_by_project(self):
        self.volume_client.quotas.get = mock.Mock(side_effect=self.volume_quotas)
        arglist = ['--volume', '--project', self.projects[0].name]
        verifylist = [('volume', True), ('project', self.projects[0].name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        ret_quotas = list(data)
        self.assertEqual(self.volume_column_header, columns)
        self.assertEqual(self.volume_reference_data, ret_quotas[0])
        self.assertEqual(1, len(ret_quotas))