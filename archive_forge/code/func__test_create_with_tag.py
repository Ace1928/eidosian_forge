import random
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def _test_create_with_tag(self, add_tags=True):
    arglist = [self._network.name]
    if add_tags:
        arglist += ['--tag', 'red', '--tag', 'blue']
    else:
        arglist += ['--no-tag']
    verifylist = [('name', self._network.name), ('enable', True), ('share', None), ('project', None), ('external', False)]
    if add_tags:
        verifylist.append(('tags', ['red', 'blue']))
    else:
        verifylist.append(('no_tag', True))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_network.assert_called_once_with(name=self._network.name, admin_state_up=True)
    if add_tags:
        self.network_client.set_tags.assert_called_once_with(self._network, tests_utils.CompareBySet(['red', 'blue']))
    else:
        self.assertFalse(self.network_client.set_tags.called)
    self.assertEqual(set(self.columns), set(columns))
    self.assertCountEqual(self.data, data)