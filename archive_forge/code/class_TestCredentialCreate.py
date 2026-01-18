from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.identity.v3 import credential
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
class TestCredentialCreate(TestCredential):
    user = identity_fakes.FakeUser.create_one_user()
    project = identity_fakes.FakeProject.create_one_project()
    columns = ('blob', 'id', 'project_id', 'type', 'user_id')

    def setUp(self):
        super(TestCredentialCreate, self).setUp()
        self.credential = identity_fakes.FakeCredential.create_one_credential(attrs={'user_id': self.user.id, 'project_id': self.project.id})
        self.credentials_mock.create.return_value = self.credential
        self.users_mock.get.return_value = self.user
        self.projects_mock.get.return_value = self.project
        self.data = (self.credential.blob, self.credential.id, self.credential.project_id, self.credential.type, self.credential.user_id)
        self.cmd = credential.CreateCredential(self.app, None)

    def test_credential_create_no_options(self):
        arglist = [self.credential.user_id, self.credential.blob]
        verifylist = [('user', self.credential.user_id), ('data', self.credential.blob)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'user': self.credential.user_id, 'type': self.credential.type, 'blob': self.credential.blob, 'project': None}
        self.credentials_mock.create.assert_called_once_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_credential_create_with_options(self):
        arglist = [self.credential.user_id, self.credential.blob, '--type', self.credential.type, '--project', self.credential.project_id]
        verifylist = [('user', self.credential.user_id), ('data', self.credential.blob), ('type', self.credential.type), ('project', self.credential.project_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'user': self.credential.user_id, 'type': self.credential.type, 'blob': self.credential.blob, 'project': self.credential.project_id}
        self.credentials_mock.create.assert_called_once_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)