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
class TestPassthruCall(TestBaremetal):

    def setUp(self):
        super(TestPassthruCall, self).setUp()
        self.cmd = baremetal_node.PassthruCallBaremetalNode(self.app, None)

    def test_passthru_call(self):
        arglist = ['node_uuid', 'heartbeat']
        verifylist = [('node', 'node_uuid'), ('method', 'heartbeat')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.vendor_passthru.assert_called_once_with('node_uuid', 'heartbeat', http_method='POST', args={})

    def test_passthru_call_http_method(self):
        arglist = ['node_uuid', 'heartbeat', '--http-method', 'PUT']
        verifylist = [('node', 'node_uuid'), ('method', 'heartbeat'), ('http_method', 'PUT')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.vendor_passthru.assert_called_once_with('node_uuid', 'heartbeat', http_method='PUT', args={})

    def test_passthru_call_args(self):
        arglist = ['node_uuid', 'heartbeat', '--arg', 'key1=value1', '--arg', 'key2=value2']
        verifylist = [('node', 'node_uuid'), ('method', 'heartbeat'), ('arg', ['key1=value1', 'key2=value2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        expected_dict = {'key1': 'value1', 'key2': 'value2'}
        self.baremetal_mock.node.vendor_passthru.assert_called_once_with('node_uuid', 'heartbeat', http_method='POST', args=expected_dict)