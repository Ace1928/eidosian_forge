from unittest import mock
from openstackclient.common import extension
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def _test_extension_list_helper(self, arglist, verifylist, expected_data, long=False):
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    if long:
        self.assertEqual(self.long_columns, columns)
    else:
        self.assertEqual(self.columns, columns)
    self.assertEqual(expected_data, tuple(data))