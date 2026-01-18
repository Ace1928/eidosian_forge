import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestChassisShow(TestChassis):

    def setUp(self):
        super(TestChassisShow, self).setUp()
        self.baremetal_mock.chassis.get.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL_CHASSIS), loaded=True)
        self.cmd = baremetal_chassis.ShowBaremetalChassis(self.app, None)

    def test_chassis_show(self):
        arglist = [baremetal_fakes.baremetal_chassis_uuid]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        args = [baremetal_fakes.baremetal_chassis_uuid]
        self.baremetal_mock.chassis.get.assert_called_with(*args, fields=None)
        collist = ('description', 'extra', 'uuid')
        self.assertEqual(collist, columns)
        self.assertNotIn('nodes', columns)
        datalist = (baremetal_fakes.baremetal_chassis_description, baremetal_fakes.baremetal_chassis_extra, baremetal_fakes.baremetal_chassis_uuid)
        self.assertEqual(datalist, tuple(data))

    def test_chassis_show_no_chassis(self):
        arglist = []
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_chassis_show_fields(self):
        uuid = baremetal_fakes.baremetal_chassis_uuid
        arglist = [uuid, '--fields', 'uuid', 'description']
        verifylist = [('chassis', uuid), ('fields', [['uuid', 'description']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = [uuid]
        fields = ['uuid', 'description']
        self.baremetal_mock.chassis.get.assert_called_with(*args, fields=fields)

    def test_chassis_show_fields_multiple(self):
        uuid = baremetal_fakes.baremetal_chassis_uuid
        arglist = [uuid, '--fields', 'uuid', 'description', '--fields', 'extra']
        verifylist = [('chassis', uuid), ('fields', [['uuid', 'description'], ['extra']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = [uuid]
        fields = ['uuid', 'description', 'extra']
        self.baremetal_mock.chassis.get.assert_called_with(*args, fields=fields)

    def test_chassis_show_invalid_fields(self):
        uuid = baremetal_fakes.baremetal_chassis_uuid
        arglist = [uuid, '--fields', 'uuid', 'invalid']
        verifylist = [('chassis', uuid), ('fields', [['uuid', 'invalid']])]
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)