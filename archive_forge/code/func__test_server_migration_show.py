from unittest import mock
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_migration
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def _test_server_migration_show(self):
    arglist = [self.server.id, '2']
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)
    self.compute_sdk_client.find_server.assert_called_with(self.server.id, ignore_missing=False)
    self.compute_sdk_client.get_server_migration.assert_called_with(self.server.id, '2', ignore_missing=False)