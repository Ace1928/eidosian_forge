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
def _test_set_tags(self, with_tags=True):
    if with_tags:
        arglist = ['--tag', 'red', '--tag', 'blue']
        verifylist = [('tags', ['red', 'blue'])]
        expected_args = ['red', 'blue', 'green']
    else:
        arglist = ['--no-tag']
        verifylist = [('no_tag', True)]
        expected_args = []
    arglist.append(self._network.name)
    verifylist.append(('network', self._network.name))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertFalse(self.network_client.update_network.called)
    self.network_client.set_tags.assert_called_once_with(self._network, tests_utils.CompareBySet(expected_args))
    self.assertIsNone(result)