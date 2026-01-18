from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import role
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestRoleShow(TestRole):

    def setUp(self):
        super(TestRoleShow, self).setUp()
        self.roles_mock.get.return_value = self.fake_role
        self.cmd = role.ShowRole(self.app, None)

    def test_service_show(self):
        arglist = [self.fake_role.name]
        verifylist = [('role', self.fake_role.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.roles_mock.get.assert_called_with(self.fake_role.name)
        collist = ('id', 'name')
        self.assertEqual(collist, columns)
        datalist = (self.fake_role.id, self.fake_role.name)
        self.assertEqual(datalist, data)