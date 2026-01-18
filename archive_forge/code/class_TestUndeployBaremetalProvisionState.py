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
class TestUndeployBaremetalProvisionState(TestBaremetal):

    def setUp(self):
        super(TestUndeployBaremetalProvisionState, self).setUp()
        self.cmd = baremetal_node.UndeployBaremetalNode(self.app, None)

    def test_undeploy_no_wait(self):
        arglist = ['node_uuid']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'deleted')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.wait_for_provision_state.assert_not_called()

    def test_undeploy_baremetal_provision_state_available_and_wait(self):
        arglist = ['node_uuid', '--wait', '15']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'deleted'), ('wait_timeout', 15)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        test_node = self.baremetal_mock.node
        test_node.wait_for_provision_state.assert_called_once_with(['node_uuid'], expected_state='available', poll_interval=10, timeout=15)

    def test_undeploy_baremetal_provision_state_default_wait(self):
        arglist = ['node_uuid', '--wait']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'deleted'), ('wait_timeout', 0)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        test_node = self.baremetal_mock.node
        test_node.wait_for_provision_state.assert_called_once_with(['node_uuid'], expected_state='available', poll_interval=10, timeout=0)