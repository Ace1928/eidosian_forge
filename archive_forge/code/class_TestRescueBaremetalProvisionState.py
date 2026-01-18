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
class TestRescueBaremetalProvisionState(TestBaremetal):

    def setUp(self):
        super(TestRescueBaremetalProvisionState, self).setUp()
        self.cmd = baremetal_node.RescueBaremetalNode(self.app, None)

    def test_rescue_baremetal_no_wait(self):
        arglist = ['node_uuid', '--rescue-password', 'supersecret']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'rescue'), ('rescue_password', 'supersecret')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_provision_state.assert_called_once_with('node_uuid', 'rescue', cleansteps=None, deploysteps=None, configdrive=None, rescue_password='supersecret')

    def test_rescue_baremetal_provision_state_rescue_and_wait(self):
        arglist = ['node_uuid', '--wait', '15', '--rescue-password', 'supersecret']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'rescue'), ('rescue_password', 'supersecret'), ('wait_timeout', 15)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        test_node = self.baremetal_mock.node
        test_node.wait_for_provision_state.assert_called_once_with(['node_uuid'], expected_state='rescue', poll_interval=10, timeout=15)

    def test_rescue_baremetal_provision_state_default_wait(self):
        arglist = ['node_uuid', '--wait', '--rescue-password', 'supersecret']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'rescue'), ('rescue_password', 'supersecret'), ('wait_timeout', 0)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        test_node = self.baremetal_mock.node
        test_node.wait_for_provision_state.assert_called_once_with(['node_uuid'], expected_state='rescue', poll_interval=10, timeout=0)

    def test_rescue_baremetal_no_rescue_password(self):
        arglist = ['node_uuid']
        verifylist = [('node', 'node_uuid'), ('provision_state', 'rescue')]
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)