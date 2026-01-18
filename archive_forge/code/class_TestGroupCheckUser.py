from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestGroupCheckUser(TestGroup):
    group = identity_fakes.FakeGroup.create_one_group()
    user = identity_fakes.FakeUser.create_one_user()

    def setUp(self):
        super(TestGroupCheckUser, self).setUp()
        self.groups_mock.get.return_value = self.group
        self.users_mock.get.return_value = self.user
        self.users_mock.check_in_group.return_value = None
        self.cmd = group.CheckUserInGroup(self.app, None)

    def test_group_check_user(self):
        arglist = [self.group.name, self.user.name]
        verifylist = [('group', self.group.name), ('user', self.user.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.users_mock.check_in_group.assert_called_once_with(self.user.id, self.group.id)
        self.assertIsNone(result)

    def test_group_check_user_server_error(self):

        def server_error(*args):
            raise ks_exc.http.InternalServerError
        self.users_mock.check_in_group.side_effect = server_error
        arglist = [self.group.name, self.user.name]
        verifylist = [('group', self.group.name), ('user', self.user.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(ks_exc.http.InternalServerError, self.cmd.take_action, parsed_args)