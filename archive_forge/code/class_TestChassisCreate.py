import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestChassisCreate(TestChassis):

    def setUp(self):
        super(TestChassisCreate, self).setUp()
        self.baremetal_mock.chassis.create.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL_CHASSIS), loaded=True)
        self.cmd = baremetal_chassis.CreateBaremetalChassis(self.app, None)
        self.arglist = []
        self.verifylist = []
        self.collist = ('description', 'extra', 'uuid')
        self.datalist = (baremetal_fakes.baremetal_chassis_description, baremetal_fakes.baremetal_chassis_extra, baremetal_fakes.baremetal_chassis_uuid)
        self.actual_kwargs = {}

    def check_with_options(self, addl_arglist, addl_verifylist, addl_kwargs):
        arglist = copy.copy(self.arglist) + addl_arglist
        verifylist = copy.copy(self.verifylist) + addl_verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        collist = copy.copy(self.collist)
        self.assertEqual(collist, columns)
        self.assertNotIn('nodes', columns)
        datalist = copy.copy(self.datalist)
        self.assertEqual(datalist, tuple(data))
        kwargs = copy.copy(self.actual_kwargs)
        kwargs.update(addl_kwargs)
        self.baremetal_mock.chassis.create.assert_called_once_with(**kwargs)

    def test_chassis_create_no_options(self):
        self.check_with_options([], [], {})

    def test_chassis_create_with_description(self):
        description = 'chassis description'
        self.check_with_options(['--description', description], [('description', description)], {'description': description})

    def test_chassis_create_with_extra(self):
        extra1 = 'arg1=val1'
        extra2 = 'arg2=val2'
        self.check_with_options(['--extra', extra1, '--extra', extra2], [('extra', [extra1, extra2])], {'extra': {'arg1': 'val1', 'arg2': 'val2'}})

    def test_chassis_create_with_uuid(self):
        uuid = baremetal_fakes.baremetal_chassis_uuid
        self.check_with_options(['--uuid', uuid], [('uuid', uuid)], {'uuid': uuid})