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
def _test_stack_environment_show(self, env):
    self.stack_client.environment = mock.MagicMock(return_value=env)
    parsed_args = self.check_parser(self.cmd, ['test-stack'], [])
    columns, outputs = self.cmd.take_action(parsed_args)
    self.assertEqual(['parameters', 'resource_registry', 'parameter_defaults'], columns)
    return (columns, outputs)