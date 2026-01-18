import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestShowBaremetalAllocation(TestBaremetalAllocation):

    def setUp(self):
        super(TestShowBaremetalAllocation, self).setUp()
        self.baremetal_mock.allocation.get.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.ALLOCATION), loaded=True)
        self.cmd = baremetal_allocation.ShowBaremetalAllocation(self.app, None)

    def test_baremetal_allocation_show(self):
        arglist = [baremetal_fakes.baremetal_uuid]
        verifylist = [('allocation', baremetal_fakes.baremetal_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.baremetal_mock.allocation.get.assert_called_once_with(baremetal_fakes.baremetal_uuid, fields=None)
        collist = ('name', 'node_uuid', 'resource_class', 'state', 'uuid')
        self.assertEqual(collist, columns)
        datalist = (baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_resource_class, baremetal_fakes.baremetal_allocation_state, baremetal_fakes.baremetal_uuid)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_allocation_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)