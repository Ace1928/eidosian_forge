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
class TestStackShow(TestStack):
    scenarios = [('table', dict(format='table')), ('shell', dict(format='shell')), ('value', dict(format='value'))]
    get_response = {'stack': {'disable_rollback': True, 'description': 'This is a\ndescription\n', 'parent': None, 'tags': None, 'stack_name': 'a', 'stack_user_project_id': '02ad9bd403d44ff9ba128cf9ce77f989', 'stack_status_reason': 'Stack UPDATE completed successfully', 'creation_time': '2015-08-04T04:46:10', 'links': [{'href': 'http://192.0.2.1:8004/v1/5dcd28/stacks/a/4af43781', 'rel': 'self'}], 'capabilities': [], 'notification_topics': [], 'updated_time': '2015-08-05T21:33:28', 'timeout_mins': None, 'stack_status': 'UPDATE_COMPLETE', 'stack_owner': None, 'parameters': {'OS::project_id': 'e0e5e140c5854c259a852621b65dcd28', 'OS::stack_id': '4af43781', 'OS::stack_name': 'a'}, 'id': '4af43781', 'outputs': [], 'template_description': 'This is a\ndescription\n'}}

    def setUp(self):
        super(TestStackShow, self).setUp()
        self.cmd = stack.ShowStack(self.app, None)
        self.stack_client.get.return_value = stacks.Stack(None, self.get_response)

    def test_stack_show(self):
        arglist = ['--format', self.format, 'my_stack']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.stack_client.get.assert_called_with(**{'stack_id': 'my_stack', 'resolve_outputs': True})

    def test_stack_show_explicit_no_resolve(self):
        arglist = ['--no-resolve-outputs', '--format', self.format, 'my_stack']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.stack_client.get.assert_called_with(**{'stack_id': 'my_stack', 'resolve_outputs': False})

    def test_stack_show_short(self):
        expected = ['id', 'stack_name', 'description', 'creation_time', 'updated_time', 'stack_status', 'stack_status_reason']
        columns, data = stack._show_stack(self.mock_client, 'my_stack', short=True)
        self.assertEqual(expected, columns)