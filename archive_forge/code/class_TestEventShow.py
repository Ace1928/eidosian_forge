import copy
from unittest import mock
import testscenarios
from heatclient import exc
from heatclient.osc.v1 import event
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import events
class TestEventShow(TestEvent):
    scenarios = [('table', dict(format='table')), ('shell', dict(format='shell')), ('value', dict(format='value'))]
    response = {'event': {'resource_name': 'my_resource', 'event_time': '2015-11-11T15:23:47Z', 'links': [], 'logical_resource_id': 'my_resource', 'resource_status': 'CREATE_FAILED', 'resource_status_reason': 'NotFound', 'physical_resource_id': 'null', 'id': '474bfdf0-a450-46ec-a78a-0c7faa404073'}}

    def setUp(self):
        super(TestEventShow, self).setUp()
        self.cmd = event.ShowEvent(self.app, None)

    def test_event_show(self):
        arglist = ['--format', self.format, 'my_stack', 'my_resource', '1234']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.event_client.get.return_value = events.Event(None, self.response)
        self.cmd.take_action(parsed_args)
        self.event_client.get.assert_called_with(**{'stack_id': 'my_stack', 'resource_name': 'my_resource', 'event_id': '1234'})

    def _test_not_found(self, error):
        arglist = ['my_stack', 'my_resource', '1234']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        ex = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn(error, str(ex))

    def test_event_show_stack_not_found(self):
        error = 'Stack not found'
        self.stack_client.get.side_effect = exc.HTTPNotFound(error)
        self._test_not_found(error)

    def test_event_show_resource_not_found(self):
        error = 'Resource not found'
        self.stack_client.get.side_effect = exc.HTTPNotFound(error)
        self._test_not_found(error)

    def test_event_show_event_not_found(self):
        error = 'Event not found'
        self.stack_client.get.side_effect = exc.HTTPNotFound(error)
        self._test_not_found(error)