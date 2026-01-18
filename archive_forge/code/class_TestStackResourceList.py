import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resources as v1_resources
class TestStackResourceList(TestResource):
    response = {'attributes': {}, 'creation_time': '2016-02-01T20:20:53', 'description': 'a resource', 'links': [{'rel': 'stack', 'href': 'http://heat.example.com:8004/my_stack/12'}], 'logical_resource_id': '1234', 'physical_resource_id': '1234', 'required_by': [], 'resource_name': 'my_resource', 'resource_status': 'CREATE_COMPLETE', 'resource_status_reason': 'state changed', 'resource_type': 'OS::Heat::None', 'updated_time': '2016-02-01T20:20:53'}
    columns = ['resource_name', 'physical_resource_id', 'resource_type', 'resource_status', 'updated_time']
    data = ['my_resource', '1234', 'OS::Heat::None', 'CREATE_COMPLETE', '2016-02-01T20:20:53']

    def setUp(self):
        super(TestStackResourceList, self).setUp()
        self.cmd = resource.ResourceList(self.app, None)
        self.resource_client.list.return_value = [v1_resources.Resource(None, self.response)]

    def test_resource_list(self):
        arglist = ['my_stack']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.resource_client.list.assert_called_with('my_stack', filters={}, with_detail=False, nested_depth=None)
        self.assertEqual(self.columns, columns)
        self.assertEqual(tuple(self.data), list(data)[0])

    def test_resource_list_not_found(self):
        arglist = ['bad_stack']
        self.resource_client.list.side_effect = heat_exc.HTTPNotFound
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)

    def test_resource_list_with_detail(self):
        arglist = ['my_stack', '--long']
        cols = copy.deepcopy(self.columns)
        cols.append('stack_name')
        out = copy.deepcopy(self.data)
        out.append('my_stack')
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.resource_client.list.assert_called_with('my_stack', filters={}, with_detail=True, nested_depth=None)
        self.assertEqual(cols, columns)
        self.assertEqual(tuple(out), list(data)[0])

    def test_resource_list_nested_depth(self):
        arglist = ['my_stack', '--nested-depth', '3']
        cols = copy.deepcopy(self.columns)
        cols.append('stack_name')
        out = copy.deepcopy(self.data)
        out.append('my_stack')
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.resource_client.list.assert_called_with('my_stack', filters={}, with_detail=False, nested_depth=3)
        self.assertEqual(cols, columns)
        self.assertEqual(tuple(out), list(data)[0])

    def test_resource_list_no_resource_name(self):
        arglist = ['my_stack']
        resp = copy.deepcopy(self.response)
        del resp['resource_name']
        cols = copy.deepcopy(self.columns)
        cols[0] = 'logical_resource_id'
        out = copy.deepcopy(self.data)
        out[1] = '1234'
        self.resource_client.list.return_value = [v1_resources.Resource(None, resp)]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.resource_client.list.assert_called_with('my_stack', filters={}, with_detail=False, nested_depth=None)
        self.assertEqual(cols, columns)

    def test_resource_list_filter(self):
        arglist = ['my_stack', '--filter', 'name=my_resource']
        out = copy.deepcopy(self.data)
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.resource_client.list.assert_called_with('my_stack', filters=dict(name='my_resource'), with_detail=False, nested_depth=None)
        self.assertEqual(tuple(out), list(data)[0])