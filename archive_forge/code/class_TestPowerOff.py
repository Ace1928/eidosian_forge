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
class TestPowerOff(TestBaremetal):

    def setUp(self):
        super(TestPowerOff, self).setUp()
        self.cmd = baremetal_node.PowerOffBaremetalNode(self.app, None)

    def test_baremetal_power_off(self):
        arglist = ['node_uuid']
        verifylist = [('nodes', ['node_uuid']), ('soft', False), ('power_timeout', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_power_state.assert_called_once_with('node_uuid', 'off', False, timeout=None)

    def test_baremetal_power_off_timeout(self):
        arglist = ['node_uuid', '--power-timeout', '2']
        verifylist = [('nodes', ['node_uuid']), ('soft', False), ('power_timeout', 2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_power_state.assert_called_once_with('node_uuid', 'off', False, timeout=2)

    def test_baremetal_soft_power_off(self):
        arglist = ['node_uuid', '--soft']
        verifylist = [('nodes', ['node_uuid']), ('soft', True), ('power_timeout', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_power_state.assert_called_once_with('node_uuid', 'off', True, timeout=None)

    def test_baremetal_soft_power_off_timeout(self):
        arglist = ['node_uuid', '--soft', '--power-timeout', '2']
        verifylist = [('nodes', ['node_uuid']), ('soft', True), ('power_timeout', 2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_power_state.assert_called_once_with('node_uuid', 'off', True, timeout=2)

    def test_baremetal_power_off_no_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_power_off_several_nodes(self):
        arglist = ['node_uuid', 'node_name']
        verifylist = [('nodes', ['node_uuid', 'node_name']), ('soft', False), ('power_timeout', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_power_state.assert_has_calls([mock.call(n, 'off', False, timeout=None) for n in arglist])