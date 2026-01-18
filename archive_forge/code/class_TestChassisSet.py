import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestChassisSet(TestChassis):

    def setUp(self):
        super(TestChassisSet, self).setUp()
        self.baremetal_mock.chassis.update.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL_CHASSIS), loaded=True)
        self.cmd = baremetal_chassis.SetBaremetalChassis(self.app, None)

    def test_chassis_set_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_chassis_set_no_property(self):
        uuid = baremetal_fakes.baremetal_chassis_uuid
        arglist = [uuid]
        verifylist = [('chassis', uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.assertFalse(self.baremetal_mock.chassis.update.called)

    def test_chassis_set_description(self):
        description = 'new description'
        uuid = baremetal_fakes.baremetal_chassis_uuid
        arglist = [uuid, '--description', 'new description']
        verifylist = [('chassis', uuid), ('description', description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.chassis.update.assert_called_once_with(uuid, [{'path': '/description', 'value': description, 'op': 'add'}])

    def test_chassis_set_extra(self):
        uuid = baremetal_fakes.baremetal_chassis_uuid
        extra = 'foo=bar'
        arglist = [uuid, '--extra', extra]
        verifylist = [('chassis', uuid), ('extra', [extra])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.chassis.update.assert_called_once_with(uuid, [{'path': '/extra/foo', 'value': 'bar', 'op': 'add'}])