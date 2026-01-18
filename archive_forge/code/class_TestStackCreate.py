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
class TestStackCreate(TestStack):
    template_path = 'heatclient/tests/test_templates/empty.yaml'
    env_path = 'heatclient/tests/unit/var/environment.json'
    defaults = {'stack_name': 'my_stack', 'disable_rollback': True, 'parameters': {}, 'template': {'heat_template_version': '2013-05-23'}, 'files': {}, 'environment': {}}

    def setUp(self):
        super(TestStackCreate, self).setUp()
        self.cmd = stack.CreateStack(self.app, None)
        self.stack_client.create.return_value = {'stack': {'id': '1234'}}
        self.stack_client.get.return_value = {'stack_status': 'create_complete'}
        self.stack_client.preview.return_value = stacks.Stack(None, {'stack': {'id', '1234'}})
        stack._authenticated_fetcher = mock.MagicMock()

    def test_stack_create_defaults(self):
        arglist = ['my_stack', '-t', self.template_path]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.stack_client.create.assert_called_with(**self.defaults)

    def test_stack_create_with_env(self):
        arglist = ['my_stack', '-t', self.template_path, '-e', self.env_path]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.assertEqual(1, self.stack_client.create.call_count)
        args = self.stack_client.create.call_args[1]
        self.assertEqual({'parameters': {}}, args.get('environment'))
        self.assertIn(self.env_path, args.get('environment_files')[0])

    def test_stack_create_rollback(self):
        arglist = ['my_stack', '-t', self.template_path, '--enable-rollback']
        kwargs = copy.deepcopy(self.defaults)
        kwargs['disable_rollback'] = False
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.stack_client.create.assert_called_with(**kwargs)

    def test_stack_create_parameters(self):
        template_path = '/'.join(self.template_path.split('/')[:-1]) + '/parameters.yaml'
        arglist = ['my_stack', '-t', template_path, '--parameter', 'p1=a', '--parameter', 'p2=6']
        kwargs = copy.deepcopy(self.defaults)
        kwargs['parameters'] = {'p1': 'a', 'p2': '6'}
        kwargs['template']['parameters'] = {'p1': {'type': 'string'}, 'p2': {'type': 'number'}}
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.stack_client.create.assert_called_with(**kwargs)

    def test_stack_create_tags(self):
        arglist = ['my_stack', '-t', self.template_path, '--tags', 'tag1,tag2']
        kwargs = copy.deepcopy(self.defaults)
        kwargs['tags'] = 'tag1,tag2'
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.stack_client.create.assert_called_with(**kwargs)

    def test_stack_create_timeout(self):
        arglist = ['my_stack', '-t', self.template_path, '--timeout', '60']
        kwargs = copy.deepcopy(self.defaults)
        kwargs['timeout_mins'] = 60
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.stack_client.create.assert_called_with(**kwargs)

    def test_stack_create_pre_create(self):
        arglist = ['my_stack', '-t', self.template_path, '--pre-create', 'a']
        kwargs = copy.deepcopy(self.defaults)
        kwargs['environment'] = {'resource_registry': {'resources': {'a': {'hooks': 'pre-create'}}}}
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.stack_client.create.assert_called_with(**kwargs)

    @mock.patch('heatclient.common.event_utils.poll_for_events', return_value=('CREATE_COMPLETE', 'Stack my_stack CREATE_COMPLETE'))
    def test_stack_create_wait(self, mock_poll):
        arglist = ['my_stack', '-t', self.template_path, '--wait']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        mock_poll.assert_called_once_with(mock.ANY, 'my_stack', action='CREATE', poll_period=5)
        self.stack_client.create.assert_called_with(**self.defaults)
        self.stack_client.get.assert_called_with(**{'stack_id': '1234', 'resolve_outputs': False})

    @mock.patch('heatclient.common.event_utils.poll_for_events', return_value=('CREATE_COMPLETE', 'Stack my_stack CREATE_COMPLETE'))
    def test_stack_create_wait_with_poll(self, mock_poll):
        arglist = ['my_stack', '-t', self.template_path, '--wait', '--poll', '10']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        mock_poll.assert_called_once_with(mock.ANY, 'my_stack', action='CREATE', poll_period=10)
        self.stack_client.create.assert_called_with(**self.defaults)
        self.stack_client.get.assert_called_with(**{'stack_id': '1234', 'resolve_outputs': False})

    @mock.patch('heatclient.common.event_utils.poll_for_events', return_value=('CREATE_FAILED', 'Stack my_stack CREATE_FAILED'))
    def test_stack_create_wait_fail(self, mock_wait):
        arglist = ['my_stack', '-t', self.template_path, '--wait']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)

    def test_stack_create_dry_run(self):
        arglist = ['my_stack', '-t', self.template_path, '--dry-run']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.stack_client.preview.assert_called_with(**self.defaults)
        self.stack_client.create.assert_not_called()