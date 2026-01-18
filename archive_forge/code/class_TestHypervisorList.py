import json
from unittest import mock
from novaclient import exceptions as nova_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import hypervisor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
class TestHypervisorList(compute_fakes.TestComputev2):

    def setUp(self):
        super().setUp()
        self.hypervisors = compute_fakes.create_hypervisors()
        self.compute_sdk_client.hypervisors.return_value = self.hypervisors
        self.columns = ('ID', 'Hypervisor Hostname', 'Hypervisor Type', 'Host IP', 'State')
        self.columns_long = ('ID', 'Hypervisor Hostname', 'Hypervisor Type', 'Host IP', 'State', 'vCPUs Used', 'vCPUs', 'Memory MB Used', 'Memory MB')
        self.data = ((self.hypervisors[0].id, self.hypervisors[0].name, self.hypervisors[0].hypervisor_type, self.hypervisors[0].host_ip, self.hypervisors[0].state), (self.hypervisors[1].id, self.hypervisors[1].name, self.hypervisors[1].hypervisor_type, self.hypervisors[1].host_ip, self.hypervisors[1].state))
        self.data_long = ((self.hypervisors[0].id, self.hypervisors[0].name, self.hypervisors[0].hypervisor_type, self.hypervisors[0].host_ip, self.hypervisors[0].state, self.hypervisors[0].vcpus_used, self.hypervisors[0].vcpus, self.hypervisors[0].memory_used, self.hypervisors[0].memory_size), (self.hypervisors[1].id, self.hypervisors[1].name, self.hypervisors[1].hypervisor_type, self.hypervisors[1].host_ip, self.hypervisors[1].state, self.hypervisors[1].vcpus_used, self.hypervisors[1].vcpus, self.hypervisors[1].memory_used, self.hypervisors[1].memory_size))
        self.cmd = hypervisor.ListHypervisor(self.app, None)

    def test_hypervisor_list_no_option(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.hypervisors.assert_called_with(details=True)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_hypervisor_list_matching_option_found(self):
        arglist = ['--matching', self.hypervisors[0].name]
        verifylist = [('matching', self.hypervisors[0].name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_sdk_client.hypervisors.return_value = [self.hypervisors[0]]
        self.data = ((self.hypervisors[0].id, self.hypervisors[0].name, self.hypervisors[1].hypervisor_type, self.hypervisors[1].host_ip, self.hypervisors[1].state),)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.hypervisors.assert_called_with(hypervisor_hostname_pattern=self.hypervisors[0].name, details=True)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_hypervisor_list_matching_option_not_found(self):
        arglist = ['--matching', 'xxx']
        verifylist = [('matching', 'xxx')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_sdk_client.hypervisors.side_effect = exceptions.NotFound(None)
        self.assertRaises(exceptions.NotFound, self.cmd.take_action, parsed_args)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_hypervisor_list_with_matching_and_pagination_options(self, sm_mock):
        arglist = ['--matching', self.hypervisors[0].name, '--limit', '1', '--marker', self.hypervisors[0].name]
        verifylist = [('matching', self.hypervisors[0].name), ('limit', 1), ('marker', self.hypervisors[0].name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--matching is not compatible with --marker or --limit', str(ex))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_hypervisor_list_long_option(self, sm_mock):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.hypervisors.assert_called_with(details=True)
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data_long, tuple(data))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_hypervisor_list_with_limit(self, sm_mock):
        arglist = ['--limit', '1']
        verifylist = [('limit', 1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.hypervisors.assert_called_with(limit=1, details=True)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_hypervisor_list_with_limit_pre_v233(self, sm_mock):
        arglist = ['--limit', '1']
        verifylist = [('limit', 1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.33 or greater is required', str(ex))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_hypervisor_list_with_marker(self, sm_mock):
        arglist = ['--marker', 'test_hyp']
        verifylist = [('marker', 'test_hyp')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.hypervisors.assert_called_with(marker='test_hyp', details=True)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_hypervisor_list_with_marker_pre_v233(self, sm_mock):
        arglist = ['--marker', 'test_hyp']
        verifylist = [('marker', 'test_hyp')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.33 or greater is required', str(ex))