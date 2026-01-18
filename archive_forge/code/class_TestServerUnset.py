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
class TestServerUnset(TestServer):

    def setUp(self):
        super(TestServerUnset, self).setUp()
        self.fake_server = self.setup_servers_mock(1)[0]
        self.cmd = server.UnsetServer(self.app, None)

    def test_server_unset_no_option(self):
        arglist = ['foo_vm']
        verifylist = [('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertNotCalled(self.servers_mock.delete_meta)
        self.assertIsNone(result)

    def test_server_unset_with_property(self):
        arglist = ['--property', 'key1', '--property', 'key2', 'foo_vm']
        verifylist = [('properties', ['key1', 'key2']), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.delete_meta.assert_called_once_with(self.fake_server, ['key1', 'key2'])
        self.assertIsNone(result)

    def test_server_unset_with_description_api_newer(self):
        self.compute_client.api_version = 2.19
        arglist = ['--description', 'foo_vm']
        verifylist = [('description', True), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(api_versions, 'APIVersion', return_value=2.19):
            result = self.cmd.take_action(parsed_args)
        self.servers_mock.update.assert_called_once_with(self.fake_server, description='')
        self.assertIsNone(result)

    def test_server_unset_with_description_api_older(self):
        self.compute_client.api_version = api_versions.APIVersion('2.18')
        arglist = ['--description', 'foo_vm']
        verifylist = [('description', True), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.19 or greater is required', str(ex))

    def test_server_unset_with_tag(self):
        self.compute_client.api_version = api_versions.APIVersion('2.26')
        arglist = ['--tag', 'tag1', '--tag', 'tag2', 'foo_vm']
        verifylist = [('tags', ['tag1', 'tag2']), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.servers_mock.delete_tag.assert_has_calls([mock.call(self.fake_server, tag='tag1'), mock.call(self.fake_server, tag='tag2')])

    def test_server_unset_with_tag_pre_v226(self):
        self.compute_client.api_version = api_versions.APIVersion('2.25')
        arglist = ['--tag', 'tag1', '--tag', 'tag2', 'foo_vm']
        verifylist = [('tags', ['tag1', 'tag2']), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.26 or greater is required', str(ex))