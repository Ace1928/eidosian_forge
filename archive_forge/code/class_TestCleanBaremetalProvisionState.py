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
class TestCleanBaremetalProvisionState(TestBaremetal):

    def setUp(self):
        super(TestCleanBaremetalProvisionState, self).setUp()
        self.cmd = baremetal_node.CleanBaremetalNode(self.app, None)

    def test_clean_no_wait(self):
        arglist = ['node_uuid', '--clean-steps', '-']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'clean'), ('clean_steps', '-')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.wait_for_provision_state.assert_not_called()

    def test_clean_baremetal_provision_state_manageable_and_wait(self):
        arglist = ['node_uuid', '--wait', '15', '--clean-steps', '-']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'clean'), ('wait_timeout', 15), ('clean_steps', '-')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        test_node = self.baremetal_mock.node
        test_node.wait_for_provision_state.assert_called_once_with(['node_uuid'], expected_state='manageable', poll_interval=10, timeout=15)

    def test_clean_baremetal_provision_state_default_wait(self):
        arglist = ['node_uuid', '--wait', '--clean-steps', '-']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'clean'), ('wait_timeout', 0), ('clean_steps', '-')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        test_node = self.baremetal_mock.node
        test_node.wait_for_provision_state.assert_called_once_with(['node_uuid'], expected_state='manageable', poll_interval=10, timeout=0)