import copy
from unittest import mock
import testscenarios
from heatclient import exc
from heatclient.osc.v1 import event
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import events
class TestEventList(TestEvent):
    defaults = {'stack_id': 'my_stack', 'resource_name': None, 'filters': {}, 'sort_dir': 'asc'}
    fields = ['resource_name', 'id', 'resource_status', 'resource_status_reason', 'event_time', 'physical_resource_id', 'logical_resource_id']

    class MockEvent(object):
        data = {'event_time': '2015-11-13T10:02:17', 'id': '1234', 'logical_resource_id': 'resource1', 'physical_resource_id': '', 'resource_name': 'resource1', 'resource_status': 'CREATE_COMPLETE', 'resource_status_reason': 'state changed', 'stack_name': 'my_stack'}

        def __getattr__(self, key):
            try:
                return self.data[key]
            except KeyError:
                raise AttributeError

    def setUp(self):
        super(TestEventList, self).setUp()
        self.cmd = event.ListEvent(self.app, None)
        self.event = self.MockEvent()
        self.event_client.list.return_value = [self.event]
        self.resource_client.list.return_value = {}

    def test_event_list_defaults(self):
        arglist = ['my_stack', '--format', 'table']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.event_client.list.assert_called_with(**self.defaults)
        self.assertEqual(self.fields, columns)

    def test_event_list_resource_nested_depth(self):
        arglist = ['my_stack', '--resource', 'my_resource', '--nested-depth', '3', '--format', 'table']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)

    def test_event_list_logical_resource_id(self):
        arglist = ['my_stack', '--format', 'table']
        del self.event.data['resource_name']
        cols = copy.deepcopy(self.fields)
        cols.pop()
        cols[0] = 'logical_resource_id'
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.event_client.list.assert_called_with(**self.defaults)
        self.assertEqual(cols, columns)
        self.event.data['resource_name'] = 'resource1'

    def test_event_list_nested_depth(self):
        arglist = ['my_stack', '--nested-depth', '3', '--format', 'table']
        kwargs = copy.deepcopy(self.defaults)
        kwargs['nested_depth'] = 3
        cols = copy.deepcopy(self.fields)
        cols[-1] = 'stack_name'
        cols.append('logical_resource_id')
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.event_client.list.assert_has_calls([mock.call(**kwargs), mock.call(**self.defaults)])
        self.assertEqual(cols, columns)

    @mock.patch('osc_lib.utils.sort_items')
    def test_event_list_sort(self, mock_sort_items):
        arglist = ['my_stack', '--sort', 'resource_name:desc', '--format', 'table']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        mock_event = self.MockEvent()
        mock_sort_items.return_value = [mock_event]
        columns, data = self.cmd.take_action(parsed_args)
        mock_sort_items.assert_called_with(mock.ANY, 'resource_name:desc')
        self.event_client.list.assert_called_with(filters={}, resource_name=None, sort_dir='desc', sort_keys=['resource_name'], stack_id='my_stack')
        self.assertEqual(self.fields, columns)

    @mock.patch('osc_lib.utils.sort_items')
    def test_event_list_sort_multiple(self, mock_sort_items):
        arglist = ['my_stack', '--sort', 'resource_name:desc', '--sort', 'id:asc', '--format', 'table']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        mock_event = self.MockEvent()
        mock_sort_items.return_value = [mock_event]
        columns, data = self.cmd.take_action(parsed_args)
        mock_sort_items.assert_called_with(mock.ANY, 'resource_name:desc,id:asc')
        self.event_client.list.assert_called_with(filters={}, resource_name=None, sort_dir='desc', sort_keys=['resource_name', 'id'], stack_id='my_stack')
        self.assertEqual(self.fields, columns)

    @mock.patch('osc_lib.utils.sort_items')
    def test_event_list_sort_default_key(self, mock_sort_items):
        arglist = ['my_stack', '--sort', ':desc', '--format', 'table']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        mock_event = self.MockEvent()
        mock_sort_items.return_value = [mock_event]
        columns, data = self.cmd.take_action(parsed_args)
        mock_sort_items.assert_called_with(mock.ANY, 'event_time:desc')
        self.event_client.list.assert_called_with(filters={}, resource_name=None, sort_dir='desc', sort_keys=[], stack_id='my_stack')
        self.assertEqual(self.fields, columns)

    @mock.patch('time.sleep')
    def test_event_list_follow(self, sleep):
        sleep.side_effect = [None, KeyboardInterrupt()]
        arglist = ['--follow', 'my_stack']
        expected = '2015-11-13 10:02:17 [resource1]: CREATE_COMPLETE  state changed\n2015-11-13 10:02:17 [resource1]: CREATE_COMPLETE  state changed\n'
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        defaults_with_marker = dict(self.defaults)
        defaults_with_marker['marker'] = '1234'
        self.event_client.list.assert_has_calls([mock.call(**self.defaults), mock.call(**defaults_with_marker)])
        self.assertEqual([], columns)
        self.assertEqual([], data)
        self.assertEqual(expected, self.fake_stdout.make_string())

    def test_event_list_log_format(self):
        arglist = ['my_stack']
        expected = '2015-11-13 10:02:17 [resource1]: CREATE_COMPLETE  state changed\n'
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.run(parsed_args)
        self.event_client.list.assert_called_with(**self.defaults)
        self.assertEqual(expected, self.fake_stdout.make_string())