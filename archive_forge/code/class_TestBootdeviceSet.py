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
class TestBootdeviceSet(TestBaremetal):

    def setUp(self):
        super(TestBootdeviceSet, self).setUp()
        self.cmd = baremetal_node.BootdeviceSetBaremetalNode(self.app, None)

    def test_bootdevice_set(self):
        arglist = ['node_uuid', 'bios']
        verifylist = [('nodes', ['node_uuid']), ('device', 'bios')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_boot_device.assert_called_once_with('node_uuid', 'bios', False)

    def test_bootdevice_set_persistent(self):
        arglist = ['node_uuid', 'bios', '--persistent']
        verifylist = [('nodes', ['node_uuid']), ('device', 'bios'), ('persistent', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_boot_device.assert_called_once_with('node_uuid', 'bios', True)

    def test_bootdevice_set_invalid_device(self):
        arglist = ['node_uuid', 'foo']
        verifylist = [('nodes', ['node_uuid']), ('device', 'foo')]
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_bootdevice_set_device_only(self):
        arglist = ['bios']
        verifylist = [('device', 'bios')]
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)