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
class TestStackList(TestStack):
    defaults = {'limit': None, 'marker': None, 'filters': {}, 'tags': None, 'tags_any': None, 'not_tags': None, 'not_tags_any': None, 'global_tenant': False, 'show_deleted': False, 'show_hidden': False}
    columns = ['ID', 'Stack Name', 'Stack Status', 'Creation Time', 'Updated Time']
    data = {'id': '1234', 'stack_name': 'my_stack', 'stack_status': 'CREATE_COMPLETE', 'creation_time': '2015-10-21T07:28:00Z', 'update_time': '2015-10-21T07:30:00Z', 'deletion_time': '2015-10-21T07:50:00Z'}
    data_with_project = copy.deepcopy(data)
    data_with_project['project'] = 'test_project'

    def setUp(self):
        super(TestStackList, self).setUp()
        self.cmd = stack.ListStack(self.app, None)
        self.stack_client.list.return_value = [stacks.Stack(None, self.data)]
        utils.get_dict_properties = mock.MagicMock(return_value='')

    def test_stack_list_defaults(self):
        arglist = []
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.stack_client.list.assert_called_with(**self.defaults)
        self.assertEqual(self.columns, columns)

    def test_stack_list_nested(self):
        kwargs = copy.deepcopy(self.defaults)
        kwargs['show_nested'] = True
        cols = copy.deepcopy(self.columns)
        cols.append('Parent')
        arglist = ['--nested']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.stack_client.list.assert_called_with(**kwargs)
        self.assertEqual(cols, columns)

    def test_stack_list_deleted(self):
        kwargs = copy.deepcopy(self.defaults)
        kwargs['show_deleted'] = True
        cols = copy.deepcopy(self.columns)
        cols.append('Deletion Time')
        arglist = ['--deleted']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.stack_client.list.assert_called_with(**kwargs)
        self.assertEqual(cols, columns)

    def test_stack_list_all_projects(self):
        self.stack_client.list.return_value = [stacks.Stack(None, self.data_with_project)]
        kwargs = copy.deepcopy(self.defaults)
        kwargs['global_tenant'] = True
        cols = copy.deepcopy(self.columns)
        cols.insert(2, 'Project')
        arglist = ['--all-projects']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.stack_client.list.assert_called_with(**kwargs)
        self.assertEqual(cols, columns)

    def test_stack_list_with_project(self):
        self.stack_client.list.return_value = [stacks.Stack(None, self.data_with_project)]
        kwargs = copy.deepcopy(self.defaults)
        cols = copy.deepcopy(self.columns)
        cols.insert(2, 'Project')
        arglist = []
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.stack_client.list.assert_called_with(**kwargs)
        self.assertEqual(cols, columns)

    def test_stack_list_long(self):
        self.stack_client.list.return_value = [stacks.Stack(None, self.data_with_project)]
        kwargs = copy.deepcopy(self.defaults)
        kwargs['global_tenant'] = True
        cols = copy.deepcopy(self.columns)
        cols.insert(2, 'Stack Owner')
        cols.insert(2, 'Project')
        arglist = ['--long']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.stack_client.list.assert_called_with(**kwargs)
        self.assertEqual(cols, columns)

    def test_stack_list_short(self):
        cols = ['ID', 'Stack Name', 'Stack Status']
        arglist = ['--short']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.stack_client.list.assert_called_with(**self.defaults)
        self.assertEqual(cols, columns)

    def test_stack_list_sort(self):
        arglist = ['--sort', 'stack_name:desc,id']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.stack_client.list.assert_called_with(**self.defaults)
        self.assertEqual(self.columns, columns)

    def test_stack_list_sort_invalid_key(self):
        arglist = ['--sort', 'bad_key']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)

    def test_stack_list_tags(self):
        kwargs = copy.deepcopy(self.defaults)
        kwargs['tags'] = 'tag1,tag2'
        arglist = ['--tags', 'tag1,tag2']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.stack_client.list.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)

    def test_stack_list_tags_mode(self):
        kwargs = copy.deepcopy(self.defaults)
        kwargs['not_tags'] = 'tag1,tag2'
        arglist = ['--tags', 'tag1,tag2', '--tag-mode', 'not']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.stack_client.list.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)

    def test_stack_list_tags_bad_mode(self):
        arglist = ['--tags', 'tag1,tag2', '--tag-mode', 'bad_mode']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)