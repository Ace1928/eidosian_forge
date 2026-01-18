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
class TestServerStop(TestServer):

    def setUp(self):
        super().setUp()
        self.cmd = server.StopServer(self.app, None)

    def test_server_stop_one_server(self):
        self.run_method_with_sdk_servers('stop_server', 1)

    def test_server_stop_multi_servers(self):
        self.run_method_with_sdk_servers('stop_server', 3)

    def test_server_start_with_all_projects(self):
        servers = self.setup_servers_mock(count=1)
        arglist = [servers[0].id, '--all-projects']
        verifylist = [('server', [servers[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_server.assert_called_once_with(servers[0].id, ignore_missing=False, details=False, all_projects=True)