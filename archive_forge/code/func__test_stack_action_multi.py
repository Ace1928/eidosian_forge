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
def _test_stack_action_multi(self, get_call_count=2):
    arglist = ['my_stack1', 'my_stack2']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, rows = self.cmd.take_action(parsed_args)
    self.assertEqual(2, self.action.call_count)
    self.assertEqual(get_call_count, self.mock_client.stacks.get.call_count)
    self.action.assert_called_with('my_stack2')
    self.mock_client.stacks.get.assert_called_with('my_stack2')
    self.assertEqual(self.columns, columns)
    self.assertEqual(2, len(rows))