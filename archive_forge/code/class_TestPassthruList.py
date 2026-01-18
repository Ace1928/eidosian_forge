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
class TestPassthruList(TestBaremetal):

    def setUp(self):
        super(TestPassthruList, self).setUp()
        self.cmd = baremetal_node.PassthruListBaremetalNode(self.app, None)
        self.baremetal_mock.node.get_vendor_passthru_methods.return_value = {'send_raw': {'require_exclusive_lock': True, 'attach': False, 'http_methods': ['POST'], 'description': '', 'async': True}, 'bmc_reset': {'require_exclusive_lock': True, 'attach': False, 'http_methods': ['POST'], 'description': '', 'async': True}}

    def test_passthru_list(self):
        arglist = ['node_uuid']
        verifylist = [('node', 'node_uuid')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        mock = self.baremetal_mock.node.get_vendor_passthru_methods
        mock.assert_called_once_with('node_uuid')