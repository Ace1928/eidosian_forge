import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestServerUnshelve(TestServer):

    def setUp(self):
        super().setUp()
        self.server = compute_fakes.create_one_sdk_server(attrs={'status': 'SHELVED'})
        self.compute_sdk_client.find_server.return_value = self.server
        self.compute_sdk_client.unshelve_server.return_value = None
        self.cmd = server.UnshelveServer(self.app, None)

    def test_unshelve(self):
        arglist = [self.server.id]
        verifylist = [('server', [self.server.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_server.assert_called_once_with(self.server.id, ignore_missing=False)
        self.compute_sdk_client.unshelve_server.assert_called_once_with(self.server.id)

    def test_unshelve_with_az(self):
        self._set_mock_microversion('2.77')
        arglist = ['--availability-zone', 'foo-az', self.server.id]
        verifylist = [('availability_zone', 'foo-az'), ('server', [self.server.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_server.assert_called_once_with(self.server.id, ignore_missing=False)
        self.compute_sdk_client.unshelve_server.assert_called_once_with(self.server.id, availability_zone='foo-az')

    def test_unshelve_with_az_pre_v277(self):
        self._set_mock_microversion('2.76')
        arglist = [self.server.id, '--availability-zone', 'foo-az']
        verifylist = [('availability_zone', 'foo-az'), ('server', [self.server.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.77 or greater is required ', str(ex))

    def test_unshelve_with_host(self):
        self._set_mock_microversion('2.91')
        arglist = ['--host', 'server1', self.server.id]
        verifylist = [('host', 'server1'), ('server', [self.server.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_server.assert_called_once_with(self.server.id, ignore_missing=False)
        self.compute_sdk_client.unshelve_server.assert_called_once_with(self.server.id, host='server1')

    def test_unshelve_with_host_pre_v291(self):
        self._set_mock_microversion('2.90')
        arglist = ['--host', 'server1', self.server.id]
        verifylist = [('host', 'server1'), ('server', [self.server.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.91 or greater is required to support the --host option', str(ex))

    def test_unshelve_with_no_az(self):
        self._set_mock_microversion('2.91')
        arglist = ['--no-availability-zone', self.server.id]
        verifylist = [('no_availability_zone', True), ('server', [self.server.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_server.assert_called_once_with(self.server.id, ignore_missing=False)
        self.compute_sdk_client.unshelve_server.assert_called_once_with(self.server.id, availability_zone=None)

    def test_unshelve_with_no_az_pre_v291(self):
        self._set_mock_microversion('2.90')
        arglist = ['--no-availability-zone', self.server.id]
        verifylist = [('no_availability_zone', True), ('server', [self.server.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.91 or greater is required to support the --no-availability-zone option', str(ex))

    def test_unshelve_with_no_az_and_az_conflict(self):
        self._set_mock_microversion('2.91')
        arglist = ['--availability-zone', 'foo-az', '--no-availability-zone', self.server.id]
        verifylist = [('availability_zone', 'foo-az'), ('no_availability_zone', True), ('server', [self.server.id])]
        ex = self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
        self.assertIn('argument --no-availability-zone: not allowed with argument --availability-zone', str(ex))

    @mock.patch.object(common_utils, 'wait_for_status', return_value=True)
    def test_unshelve_with_wait(self, mock_wait_for_status):
        arglist = ['--wait', self.server.name]
        verifylist = [('server', [self.server.name]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.compute_sdk_client.find_server.assert_called_with(self.server.name, ignore_missing=False)
        self.compute_sdk_client.unshelve_server.assert_called_with(self.server.id)
        mock_wait_for_status.assert_called_once_with(self.compute_sdk_client.get_server, self.server.id, callback=mock.ANY, success_status=('active', 'shutoff'))