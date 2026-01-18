import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestShowBaremetalPort(TestBaremetalPort):

    def setUp(self):
        super(TestShowBaremetalPort, self).setUp()
        self.baremetal_mock.port.get.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL_PORT), loaded=True)
        self.baremetal_mock.port.get_by_address.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL_PORT), loaded=True)
        self.cmd = baremetal_port.ShowBaremetalPort(self.app, None)

    def test_baremetal_port_show(self):
        arglist = ['zzz-zzzzzz-zzzz']
        verifylist = [('port', baremetal_fakes.baremetal_port_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        args = ['zzz-zzzzzz-zzzz']
        self.baremetal_mock.port.get.assert_called_with(*args, fields=None)
        collist = ('address', 'extra', 'node_uuid', 'uuid')
        self.assertEqual(collist, columns)
        datalist = (baremetal_fakes.baremetal_port_address, baremetal_fakes.baremetal_port_extra, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_port_uuid)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_port_show_address(self):
        arglist = ['--address', baremetal_fakes.baremetal_port_address]
        verifylist = [('address', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'AA:BB:CC:DD:EE:FF'}
        self.baremetal_mock.port.get_by_address.assert_called_with(*args, fields=None)

    def test_baremetal_port_show_no_port(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)