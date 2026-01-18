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
class TestServerAddPort(TestServer):

    def setUp(self):
        super(TestServerAddPort, self).setUp()
        self.cmd = server.AddPort(self.app, None)
        self.methods = {'interface_attach': None}
        self.find_port = mock.Mock()
        self.app.client_manager.network.find_port = self.find_port

    def _test_server_add_port(self, port_id):
        servers = self.setup_sdk_servers_mock(count=1)
        port = 'fake-port'
        arglist = [servers[0].id, port]
        verifylist = [('server', servers[0].id), ('port', port)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_server_interface.assert_called_once_with(servers[0], port_id=port_id)
        self.assertIsNone(result)

    def test_server_add_port(self):
        self._test_server_add_port(self.find_port.return_value.id)
        self.find_port.assert_called_once_with('fake-port', ignore_missing=False)

    def test_server_add_port_no_neutron(self):
        self.app.client_manager.network_endpoint_enabled = False
        self._test_server_add_port('fake-port')
        self.find_port.assert_not_called()

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_server_add_port_with_tag(self, sm_mock):
        servers = self.setup_sdk_servers_mock(count=1)
        self.find_port.return_value.id = 'fake-port'
        arglist = [servers[0].id, 'fake-port', '--tag', 'tag1']
        verifylist = [('server', servers[0].id), ('port', 'fake-port'), ('tag', 'tag1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.compute_sdk_client.create_server_interface.assert_called_once_with(servers[0], port_id='fake-port', tag='tag1')

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_server_add_port_with_tag_pre_v249(self, sm_mock):
        servers = self.setup_servers_mock(count=1)
        self.find_port.return_value.id = 'fake-port'
        arglist = [servers[0].id, 'fake-port', '--tag', 'tag1']
        verifylist = [('server', servers[0].id), ('port', 'fake-port'), ('tag', 'tag1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.49 or greater is required', str(ex))