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
@mock.patch('openstackclient.api.compute_v2.APIv2.security_group_find')
class TestServerRemoveSecurityGroup(TestServer):

    def setUp(self):
        super(TestServerRemoveSecurityGroup, self).setUp()
        self.security_group = compute_fakes.create_one_security_group()
        attrs = {'security_groups': [{'name': self.security_group['id']}]}
        methods = {'remove_security_group': None}
        self.server = compute_fakes.create_one_server(attrs=attrs, methods=methods)
        self.servers_mock.get.return_value = self.server
        self.cmd = server.RemoveServerSecurityGroup(self.app, None)

    def test_server_remove_security_group(self, sg_find_mock):
        sg_find_mock.return_value = self.security_group
        arglist = [self.server.id, self.security_group['id']]
        verifylist = [('server', self.server.id), ('group', self.security_group['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        sg_find_mock.assert_called_with(self.security_group['id'])
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.remove_security_group.assert_called_with(self.security_group['id'])
        self.assertIsNone(result)