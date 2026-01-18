from unittest import mock
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import server_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestServerGroupShow(TestServerGroup):

    def setUp(self):
        super().setUp()
        self.compute_sdk_client.find_server_group.return_value = self.fake_server_group
        self.cmd = server_group.ShowServerGroup(self.app, None)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_server_group_show(self, sm_mock):
        arglist = ['affinity_group']
        verifylist = [('server_group', 'affinity_group')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)