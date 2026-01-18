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
class TestEnvironmentStackShow(TestStack):
    SAMPLE_ENV = {'parameters': {'p1': 'v1'}, 'resource_registry': {'resources': {'r1': 't1'}}, 'parameter_defaults': {'p1': 'v_default'}}

    def setUp(self):
        super(TestEnvironmentStackShow, self).setUp()
        self.cmd = stack.EnvironmentShowStack(self.app, None)

    def test_stack_environment_show(self):
        columns, outputs = self._test_stack_environment_show(self.SAMPLE_ENV)
        self.assertEqual([{'p1': 'v1'}, {'resources': {'r1': 't1'}}, {'p1': 'v_default'}], outputs)

    def test_stack_environment_show_no_parameters(self):
        sample_env = copy.deepcopy(self.SAMPLE_ENV)
        sample_env['parameters'] = {}
        columns, outputs = self._test_stack_environment_show(sample_env)
        self.assertEqual([{}, {'resources': {'r1': 't1'}}, {'p1': 'v_default'}], outputs)

    def test_stack_environment_show_no_registry(self):
        sample_env = copy.deepcopy(self.SAMPLE_ENV)
        sample_env['resource_registry'] = {'resources': {}}
        columns, outputs = self._test_stack_environment_show(sample_env)
        self.assertEqual([{'p1': 'v1'}, {'resources': {}}, {'p1': 'v_default'}], outputs)

    def test_stack_environment_show_no_param_defaults(self):
        sample_env = copy.deepcopy(self.SAMPLE_ENV)
        sample_env['parameter_defaults'] = {}
        columns, outputs = self._test_stack_environment_show(sample_env)
        self.assertEqual([{'p1': 'v1'}, {'resources': {'r1': 't1'}}, {}], outputs)

    def _test_stack_environment_show(self, env):
        self.stack_client.environment = mock.MagicMock(return_value=env)
        parsed_args = self.check_parser(self.cmd, ['test-stack'], [])
        columns, outputs = self.cmd.take_action(parsed_args)
        self.assertEqual(['parameters', 'resource_registry', 'parameter_defaults'], columns)
        return (columns, outputs)