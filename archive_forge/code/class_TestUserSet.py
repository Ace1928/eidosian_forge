from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestUserSet(TestUser):

    def setUp(self):
        super(TestUserSet, self).setUp()
        self.projects_mock.get.return_value = self.fake_project
        self.users_mock.get.return_value = self.fake_user
        self.cmd = user.SetUser(self.app, None)

    def test_user_set_no_options(self):
        arglist = [self.fake_user.name]
        verifylist = [('name', None), ('password', None), ('email', None), ('project', None), ('enable', False), ('disable', False), ('user', self.fake_user.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)

    def test_user_set_unexist_user(self):
        arglist = ['unexist-user']
        verifylist = [('name', None), ('password', None), ('email', None), ('project', None), ('enable', False), ('disable', False), ('user', 'unexist-user')]
        self.users_mock.get.side_effect = exceptions.NotFound(None)
        self.users_mock.find.side_effect = exceptions.NotFound(None)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_user_set_name(self):
        arglist = ['--name', 'qwerty', self.fake_user.name]
        verifylist = [('name', 'qwerty'), ('password', None), ('email', None), ('project', None), ('enable', False), ('disable', False), ('user', self.fake_user.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True, 'name': 'qwerty'}
        self.users_mock.update.assert_called_with(self.fake_user.id, **kwargs)
        self.assertIsNone(result)

    def test_user_set_password(self):
        arglist = ['--password', 'secret', self.fake_user.name]
        verifylist = [('name', None), ('password', 'secret'), ('password_prompt', False), ('email', None), ('project', None), ('enable', False), ('disable', False), ('user', self.fake_user.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.users_mock.update_password.assert_called_with(self.fake_user.id, 'secret')
        self.assertIsNone(result)

    def test_user_set_password_prompt(self):
        arglist = ['--password-prompt', self.fake_user.name]
        verifylist = [('name', None), ('password', None), ('password_prompt', True), ('email', None), ('project', None), ('enable', False), ('disable', False), ('user', self.fake_user.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        mocker = mock.Mock()
        mocker.return_value = 'abc123'
        with mock.patch('osc_lib.utils.get_password', mocker):
            result = self.cmd.take_action(parsed_args)
        self.users_mock.update_password.assert_called_with(self.fake_user.id, 'abc123')
        self.assertIsNone(result)

    def test_user_set_email(self):
        arglist = ['--email', 'barney@example.com', self.fake_user.name]
        verifylist = [('name', None), ('password', None), ('email', 'barney@example.com'), ('project', None), ('enable', False), ('disable', False), ('user', self.fake_user.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'email': 'barney@example.com', 'enabled': True}
        self.users_mock.update.assert_called_with(self.fake_user.id, **kwargs)
        self.assertIsNone(result)

    def test_user_set_project(self):
        arglist = ['--project', self.fake_project.id, self.fake_user.name]
        verifylist = [('name', None), ('password', None), ('email', None), ('project', self.fake_project.id), ('enable', False), ('disable', False), ('user', self.fake_user.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.users_mock.update_tenant.assert_called_with(self.fake_user.id, self.fake_project.id)
        self.assertIsNone(result)

    def test_user_set_enable(self):
        arglist = ['--enable', self.fake_user.name]
        verifylist = [('name', None), ('password', None), ('email', None), ('project', None), ('enable', True), ('disable', False), ('user', self.fake_user.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True}
        self.users_mock.update.assert_called_with(self.fake_user.id, **kwargs)
        self.assertIsNone(result)

    def test_user_set_disable(self):
        arglist = ['--disable', self.fake_user.name]
        verifylist = [('name', None), ('password', None), ('email', None), ('project', None), ('enable', False), ('disable', True), ('user', self.fake_user.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': False}
        self.users_mock.update.assert_called_with(self.fake_user.id, **kwargs)
        self.assertIsNone(result)