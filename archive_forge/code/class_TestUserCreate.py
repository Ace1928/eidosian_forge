from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestUserCreate(TestUser):
    fake_project_c = identity_fakes.FakeProject.create_one_project()
    attr = {'tenantId': fake_project_c.id}
    fake_user_c = identity_fakes.FakeUser.create_one_user(attr)
    columns = ('email', 'enabled', 'id', 'name', 'project_id')
    datalist = (fake_user_c.email, True, fake_user_c.id, fake_user_c.name, fake_project_c.id)

    def setUp(self):
        super(TestUserCreate, self).setUp()
        self.projects_mock.get.return_value = self.fake_project_c
        self.users_mock.create.return_value = self.fake_user_c
        self.cmd = user.CreateUser(self.app, None)

    def test_user_create_no_options(self):
        arglist = [self.fake_user_c.name]
        verifylist = [('enable', False), ('disable', False), ('name', self.fake_user_c.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True, 'tenant_id': None}
        self.users_mock.create.assert_called_with(self.fake_user_c.name, None, None, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_user_create_password(self):
        arglist = ['--password', 'secret', self.fake_user_c.name]
        verifylist = [('name', self.fake_user_c.name), ('password_prompt', False), ('password', 'secret')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True, 'tenant_id': None}
        self.users_mock.create.assert_called_with(self.fake_user_c.name, 'secret', None, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_user_create_password_prompt(self):
        arglist = ['--password-prompt', self.fake_user_c.name]
        verifylist = [('name', self.fake_user_c.name), ('password_prompt', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        mocker = mock.Mock()
        mocker.return_value = 'abc123'
        with mock.patch('osc_lib.utils.get_password', mocker):
            columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True, 'tenant_id': None}
        self.users_mock.create.assert_called_with(self.fake_user_c.name, 'abc123', None, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_user_create_email(self):
        arglist = ['--email', 'barney@example.com', self.fake_user_c.name]
        verifylist = [('name', self.fake_user_c.name), ('email', 'barney@example.com')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True, 'tenant_id': None}
        self.users_mock.create.assert_called_with(self.fake_user_c.name, None, 'barney@example.com', **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_user_create_project(self):
        self.projects_mock.get.return_value = self.fake_project_c
        attr = {'tenantId': self.fake_project_c.id}
        user_2 = identity_fakes.FakeUser.create_one_user(attr)
        self.users_mock.create.return_value = user_2
        arglist = ['--project', self.fake_project_c.name, user_2.name]
        verifylist = [('name', user_2.name), ('project', self.fake_project_c.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True, 'tenant_id': self.fake_project_c.id}
        self.users_mock.create.assert_called_with(user_2.name, None, None, **kwargs)
        self.assertEqual(self.columns, columns)
        datalist = (user_2.email, True, user_2.id, user_2.name, self.fake_project_c.id)
        self.assertEqual(datalist, data)

    def test_user_create_enable(self):
        arglist = ['--enable', self.fake_user_c.name]
        verifylist = [('name', self.fake_user_c.name), ('enable', True), ('disable', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True, 'tenant_id': None}
        self.users_mock.create.assert_called_with(self.fake_user_c.name, None, None, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_user_create_disable(self):
        arglist = ['--disable', self.fake_user_c.name]
        verifylist = [('name', self.fake_user_c.name), ('enable', False), ('disable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': False, 'tenant_id': None}
        self.users_mock.create.assert_called_with(self.fake_user_c.name, None, None, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_user_create_or_show_exists(self):

        def _raise_conflict(*args, **kwargs):
            raise ks_exc.Conflict(None)
        self.users_mock.create.side_effect = _raise_conflict
        self.users_mock.get.return_value = self.fake_user_c
        arglist = ['--or-show', self.fake_user_c.name]
        verifylist = [('name', self.fake_user_c.name), ('or_show', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.users_mock.get.assert_called_with(self.fake_user_c.name)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_user_create_or_show_not_exists(self):
        arglist = ['--or-show', self.fake_user_c.name]
        verifylist = [('name', self.fake_user_c.name), ('or_show', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True, 'tenant_id': None}
        self.users_mock.create.assert_called_with(self.fake_user_c.name, None, None, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)