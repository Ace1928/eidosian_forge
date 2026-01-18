from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def _test_create_with_ha_options(self, option, ha):
    arglist = [option, self.new_router.name]
    verifylist = [('name', self.new_router.name), ('enable', True), ('distributed', False), ('ha', ha), ('no_ha', not ha)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_router.assert_called_once_with(**{'admin_state_up': True, 'name': self.new_router.name, 'ha': ha})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)