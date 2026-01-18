import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
class TestNodeHistoryEventGet(TestBaremetal):

    def setUp(self):
        super(TestNodeHistoryEventGet, self).setUp()
        self.baremetal_mock.node.get_history_event.return_value = baremetal_fakes.NODE_HISTORY[0]
        self.cmd = baremetal_node.NodeHistoryEventGet(self.app, None)

    def test_baremetal_node_history_list(self):
        arglist = ['node_uuid', 'event_uuid']
        verifylist = [('node', 'node_uuid'), ('event', 'event_uuid')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.get_history_event.assert_called_once_with('node_uuid', 'event_uuid')
        expected_columns = ('conductor', 'created_at', 'event', 'event_type', 'severity', 'user', 'uuid')
        expected_data = ('lap-conductor', 'time', 'meow', 'purring', 'info', '0191', 'abcdef1')
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, tuple(data))