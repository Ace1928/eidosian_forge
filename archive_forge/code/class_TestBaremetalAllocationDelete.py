import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalAllocationDelete(TestBaremetalAllocation):

    def setUp(self):
        super(TestBaremetalAllocationDelete, self).setUp()
        self.baremetal_mock.allocation.get.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.ALLOCATION), loaded=True)
        self.cmd = baremetal_allocation.DeleteBaremetalAllocation(self.app, None)

    def test_baremetal_allocation_delete(self):
        arglist = [baremetal_fakes.baremetal_uuid]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.allocation.delete.assert_called_once_with(baremetal_fakes.baremetal_uuid)

    def test_baremetal_allocation_delete_multiple(self):
        arglist = [baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_name]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.allocation.delete.assert_has_calls([mock.call(x) for x in arglist])
        self.assertEqual(2, self.baremetal_mock.allocation.delete.call_count)

    def test_baremetal_allocation_delete_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)