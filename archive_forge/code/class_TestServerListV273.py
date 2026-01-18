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
class TestServerListV273(_TestServerList):
    columns = ('ID', 'Name', 'Status', 'Networks', 'Image', 'Flavor')
    columns_long = ('ID', 'Name', 'Status', 'Task State', 'Power State', 'Networks', 'Image Name', 'Image ID', 'Flavor', 'Availability Zone', 'Host', 'Properties')

    def setUp(self):
        super(TestServerListV273, self).setUp()
        self.attrs['flavor'] = {'vcpus': self.flavor.vcpus, 'ram': self.flavor.ram, 'disk': self.flavor.disk, 'ephemeral': self.flavor.ephemeral, 'swap': self.flavor.swap, 'original_name': self.flavor.name, 'extra_specs': self.flavor.extra_specs}
        self.servers = self.setup_sdk_servers_mock(3)
        self.compute_sdk_client.servers.return_value = self.servers
        Image = collections.namedtuple('Image', 'id name')
        self.image_client.images.return_value = [Image(id=s.image['id'], name=self.image.name) for s in self.servers if s.image]
        self.compute_sdk_client.flavors = mock.NonCallableMock()
        self.data = tuple(((s.id, s.name, s.status, server.AddressesColumn(s.addresses), self.image.name if s.image else server.IMAGE_STRING_FOR_BFV, self.flavor.name) for s in self.servers))

    def test_server_list_with_locked_pre_v273(self):
        arglist = ['--locked']
        verifylist = [('locked', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.73 or greater is required', str(ex))

    def test_server_list_with_locked(self):
        self._set_mock_microversion('2.73')
        arglist = ['--locked']
        verifylist = [('locked', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['locked'] = True
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, tuple(data))

    def test_server_list_with_unlocked_v273(self):
        self._set_mock_microversion('2.73')
        arglist = ['--unlocked']
        verifylist = [('unlocked', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['locked'] = False
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, tuple(data))

    def test_server_list_with_locked_and_unlocked(self):
        self._set_mock_microversion('2.73')
        arglist = ['--locked', '--unlocked']
        verifylist = [('locked', True), ('unlocked', True)]
        ex = self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
        self.assertIn('Argument parse failed', str(ex))

    def test_server_list_with_changes_before(self):
        self._set_mock_microversion('2.66')
        arglist = ['--changes-before', '2016-03-05T06:27:59Z', '--deleted']
        verifylist = [('changes_before', '2016-03-05T06:27:59Z'), ('deleted', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.kwargs['changes-before'] = '2016-03-05T06:27:59Z'
        self.kwargs['deleted'] = True
        self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, tuple(data))

    @mock.patch.object(iso8601, 'parse_date', side_effect=iso8601.ParseError)
    def test_server_list_with_invalid_changes_before(self, mock_parse_isotime):
        self._set_mock_microversion('2.66')
        arglist = ['--changes-before', 'Invalid time value']
        verifylist = [('changes_before', 'Invalid time value')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('Invalid changes-before value: Invalid time value', str(e))
        mock_parse_isotime.assert_called_once_with('Invalid time value')

    def test_server_with_changes_before_pre_v266(self):
        self._set_mock_microversion('2.65')
        arglist = ['--changes-before', '2016-03-05T06:27:59Z', '--deleted']
        verifylist = [('changes_before', '2016-03-05T06:27:59Z'), ('deleted', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_server_list_v269_with_partial_constructs(self):
        self._set_mock_microversion('2.69')
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        server_dict = {'id': 'server-id-95a56bfc4xxxxxx28d7e418bfd97813a', 'status': 'UNKNOWN', 'tenant_id': '6f70656e737461636b20342065766572', 'created': '2018-12-03T21:06:18Z', 'links': [{'href': 'http://fake/v2.1/', 'rel': 'self'}, {'href': 'http://fake', 'rel': 'bookmark'}], 'networks': {}}
        fake_server = compute_fakes.fakes.FakeResource(info=server_dict)
        self.servers.append(fake_server)
        columns, data = self.cmd.take_action(parsed_args)
        next(data)
        next(data)
        next(data)
        partial_server = next(data)
        expected_row = ('server-id-95a56bfc4xxxxxx28d7e418bfd97813a', '', 'UNKNOWN', server.AddressesColumn(''), '', '')
        self.assertEqual(expected_row, partial_server)