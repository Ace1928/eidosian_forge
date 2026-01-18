import copy
import io
from unittest import mock
from osc_lib import exceptions as exc
from osc_lib import utils
import testscenarios
import yaml
from heatclient.common import template_format
from heatclient import exc as heat_exc
from heatclient.osc.v1 import stack
from heatclient.tests import inline_templates
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import events
from heatclient.v1 import resources
from heatclient.v1 import stacks
def _test_stack_cancel_no_rollback(self, call_count):
    self.action = self.mock_client.actions.cancel_without_rollback
    arglist = ['my_stack', '--no-rollback']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, rows = self.cmd.take_action(parsed_args)
    self.action.assert_called_once_with('my_stack')
    self.mock_client.stacks.get.assert_called_with('my_stack')
    self.assertEqual(call_count, self.mock_client.stacks.get.call_count)
    self.assertEqual(self.columns, columns)
    self.assertEqual(1, len(rows))