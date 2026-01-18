import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestChassisList(TestChassis):

    def setUp(self):
        super(TestChassisList, self).setUp()
        self.baremetal_mock.chassis.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL_CHASSIS), loaded=True)]
        self.cmd = baremetal_chassis.ListBaremetalChassis(self.app, None)

    def test_chassis_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.chassis.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Description')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_chassis_uuid, baremetal_fakes.baremetal_chassis_description),)
        self.assertEqual(datalist, tuple(data))

    def test_chassis_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'detail': True, 'marker': None, 'limit': None}
        self.baremetal_mock.chassis.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Description', 'Created At', 'Updated At', 'Extra')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_chassis_uuid, baremetal_fakes.baremetal_chassis_description, '', '', baremetal_fakes.baremetal_chassis_extra),)
        self.assertEqual(datalist, tuple(data))

    def test_chassis_list_fields(self):
        arglist = ['--fields', 'uuid', 'extra']
        verifylist = [('fields', [['uuid', 'extra']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'extra')}
        self.baremetal_mock.chassis.list.assert_called_with(**kwargs)

    def test_chassis_list_fields_multiple(self):
        arglist = ['--fields', 'uuid', 'description', '--fields', 'extra']
        verifylist = [('fields', [['uuid', 'description'], ['extra']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'description', 'extra')}
        self.baremetal_mock.chassis.list.assert_called_with(**kwargs)

    def test_chassis_list_invalid_fields(self):
        arglist = ['--fields', 'uuid', 'invalid']
        verifylist = [('fields', [['uuid', 'invalid']])]
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_chassis_list_long_and_fields(self):
        arglist = ['--long', '--fields', 'uuid', 'invalid']
        verifylist = [('long', True), ('fields', [['uuid', 'invalid']])]
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)