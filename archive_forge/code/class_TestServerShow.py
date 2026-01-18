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
class TestServerShow(TestServer):

    def setUp(self):
        super(TestServerShow, self).setUp()
        self.image = image_fakes.create_one_image()
        self.flavor = compute_fakes.create_one_flavor()
        self.topology = {'nodes': [{'vcpu_set': [0, 1]}, {'vcpu_set': [2, 3]}], 'pagesize_kb': None}
        server_info = {'image': {'id': self.image.id}, 'flavor': {'id': self.flavor.id}, 'tenant_id': 'tenant-id-xxx', 'addresses': {'public': ['10.20.30.40', '2001:db8::f']}}
        self.compute_sdk_client.get_server_diagnostics.return_value = {'test': 'test'}
        server_method = {'fetch_topology': self.topology}
        self.server = compute_fakes.create_one_server(attrs=server_info, methods=server_method)
        self.compute_sdk_client.get_server.return_value = self.server
        self.image_client.get_image.return_value = self.image
        self.flavors_mock.get.return_value = self.flavor
        self.cmd = server.ShowServer(self.app, None)
        self.columns = ('OS-EXT-STS:power_state', 'addresses', 'flavor', 'id', 'image', 'name', 'project_id', 'properties')
        self.data = (server.PowerStateColumn(getattr(self.server, 'OS-EXT-STS:power_state')), self.flavor.name + ' (' + self.flavor.id + ')', self.server.id, self.image.name + ' (' + self.image.id + ')', self.server.name, server.AddressesColumn({'public': ['10.20.30.40', '2001:db8::f']}), 'tenant-id-xxx', format_columns.DictColumn({}))

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show(self):
        arglist = [self.server.name]
        verifylist = [('diagnostics', False), ('topology', False), ('server', self.server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_show_embedded_flavor(self):
        arglist = [self.server.name]
        verifylist = [('diagnostics', False), ('topology', False), ('server', self.server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.server.info['flavor'] = {'ephemeral': 0, 'ram': 512, 'original_name': 'm1.tiny', 'vcpus': 1, 'extra_specs': {}, 'swap': 0, 'disk': 1}
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertIn('original_name', data[2]._value)

    def test_show_diagnostics(self):
        arglist = ['--diagnostics', self.server.name]
        verifylist = [('diagnostics', True), ('topology', False), ('server', self.server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(('test',), columns)
        self.assertEqual(('test',), data)

    def test_show_topology(self):
        self._set_mock_microversion('2.78')
        arglist = ['--topology', self.server.name]
        verifylist = [('diagnostics', False), ('topology', True), ('server', self.server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.columns += ('topology',)
        self.data += (format_columns.DictColumn(self.topology),)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_show_topology_pre_v278(self):
        self._set_mock_microversion('2.77')
        arglist = ['--topology', self.server.name]
        verifylist = [('diagnostics', False), ('topology', True), ('server', self.server.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)