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
class TestServerDumpCreate(TestServer):

    def setUp(self):
        super().setUp()
        self.cmd = server.CreateServerDump(self.app, None)

    def run_test_server_dump(self, server_count):
        servers = self.setup_sdk_servers_mock(server_count)
        arglist = []
        verifylist = []
        for s in servers:
            arglist.append(s.id)
        verifylist = [('server', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        for s in servers:
            s.trigger_crash_dump.assert_called_once_with(self.compute_sdk_client)

    def test_server_dump_one_server(self):
        self.run_test_server_dump(1)

    def test_server_dump_multi_servers(self):
        self.run_test_server_dump(3)