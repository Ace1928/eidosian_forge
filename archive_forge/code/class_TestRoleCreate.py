from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import role
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestRoleCreate(TestRole):
    fake_role_c = identity_fakes.FakeRole.create_one_role()
    columns = ('id', 'name')
    datalist = (fake_role_c.id, fake_role_c.name)

    def setUp(self):
        super(TestRoleCreate, self).setUp()
        self.roles_mock.create.return_value = self.fake_role_c
        self.cmd = role.CreateRole(self.app, None)

    def test_role_create_no_options(self):
        arglist = [self.fake_role_c.name]
        verifylist = [('role_name', self.fake_role_c.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.roles_mock.create.assert_called_with(self.fake_role_c.name)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_role_create_or_show_exists(self):

        def _raise_conflict(*args, **kwargs):
            raise ks_exc.Conflict(None)
        self.roles_mock.create.side_effect = _raise_conflict
        self.roles_mock.get.return_value = self.fake_role_c
        arglist = ['--or-show', self.fake_role_c.name]
        verifylist = [('role_name', self.fake_role_c.name), ('or_show', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.roles_mock.get.assert_called_with(self.fake_role_c.name)
        self.roles_mock.create.assert_called_with(self.fake_role_c.name)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_role_create_or_show_not_exists(self):
        arglist = ['--or-show', self.fake_role_c.name]
        verifylist = [('role_name', self.fake_role_c.name), ('or_show', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.roles_mock.create.assert_called_with(self.fake_role_c.name)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)