from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def _test_create_with_numa_affinity_policy(self, policy):
    arglist = ['--numa-policy-%s' % policy, self._port.id]
    verifylist = [('numa_policy_%s' % policy, True), ('port', self._port.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.network_client.update_port.assert_called_once_with(self._port, **{'numa_affinity_policy': policy})