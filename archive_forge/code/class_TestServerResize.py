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
class TestServerResize(TestServer):

    def setUp(self):
        super(TestServerResize, self).setUp()
        self.server = compute_fakes.create_one_server()
        self.servers_mock.get.return_value = self.server
        self.servers_mock.resize.return_value = None
        self.servers_mock.confirm_resize.return_value = None
        self.servers_mock.revert_resize.return_value = None
        self.flavors_get_return_value = compute_fakes.create_one_flavor()
        self.flavors_mock.get.return_value = self.flavors_get_return_value
        self.cmd = server.ResizeServer(self.app, None)

    def test_server_resize_no_options(self):
        arglist = [self.server.id]
        verifylist = [('confirm', False), ('revert', False), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.assertNotCalled(self.servers_mock.resize)
        self.assertNotCalled(self.servers_mock.confirm_resize)
        self.assertNotCalled(self.servers_mock.revert_resize)
        self.assertIsNone(result)

    def test_server_resize(self):
        arglist = ['--flavor', self.flavors_get_return_value.id, self.server.id]
        verifylist = [('flavor', self.flavors_get_return_value.id), ('confirm', False), ('revert', False), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.flavors_mock.get.assert_called_with(self.flavors_get_return_value.id)
        self.servers_mock.resize.assert_called_with(self.server, self.flavors_get_return_value)
        self.assertNotCalled(self.servers_mock.confirm_resize)
        self.assertNotCalled(self.servers_mock.revert_resize)
        self.assertIsNone(result)

    def test_server_resize_confirm(self):
        arglist = ['--confirm', self.server.id]
        verifylist = [('confirm', True), ('revert', False), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(self.cmd.log, 'warning') as mock_warning:
            result = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.assertNotCalled(self.servers_mock.resize)
        self.servers_mock.confirm_resize.assert_called_with(self.server)
        self.assertNotCalled(self.servers_mock.revert_resize)
        self.assertIsNone(result)
        mock_warning.assert_called_once()
        self.assertIn('The --confirm option has been deprecated.', str(mock_warning.call_args[0][0]))

    def test_server_resize_revert(self):
        arglist = ['--revert', self.server.id]
        verifylist = [('confirm', False), ('revert', True), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(self.cmd.log, 'warning') as mock_warning:
            result = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.assertNotCalled(self.servers_mock.resize)
        self.assertNotCalled(self.servers_mock.confirm_resize)
        self.servers_mock.revert_resize.assert_called_with(self.server)
        self.assertIsNone(result)
        mock_warning.assert_called_once()
        self.assertIn('The --revert option has been deprecated.', str(mock_warning.call_args[0][0]))

    @mock.patch.object(common_utils, 'wait_for_status', return_value=True)
    def test_server_resize_with_wait_ok(self, mock_wait_for_status):
        arglist = ['--flavor', self.flavors_get_return_value.id, '--wait', self.server.id]
        verifylist = [('flavor', self.flavors_get_return_value.id), ('confirm', False), ('revert', False), ('wait', True), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        kwargs = dict(success_status=['active', 'verify_resize'])
        mock_wait_for_status.assert_called_once_with(self.servers_mock.get, self.server.id, callback=mock.ANY, **kwargs)
        self.servers_mock.resize.assert_called_with(self.server, self.flavors_get_return_value)
        self.assertNotCalled(self.servers_mock.confirm_resize)
        self.assertNotCalled(self.servers_mock.revert_resize)

    @mock.patch.object(common_utils, 'wait_for_status', return_value=False)
    def test_server_resize_with_wait_fails(self, mock_wait_for_status):
        arglist = ['--flavor', self.flavors_get_return_value.id, '--wait', self.server.id]
        verifylist = [('flavor', self.flavors_get_return_value.id), ('confirm', False), ('revert', False), ('wait', True), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        kwargs = dict(success_status=['active', 'verify_resize'])
        mock_wait_for_status.assert_called_once_with(self.servers_mock.get, self.server.id, callback=mock.ANY, **kwargs)
        self.servers_mock.resize.assert_called_with(self.server, self.flavors_get_return_value)