from unittest import mock
from openstackclient.identity.v2_0 import token
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestTokenIssue(TestToken):

    def setUp(self):
        super(TestTokenIssue, self).setUp()
        self.cmd = token.IssueToken(self.app, None)

    def test_token_issue(self):
        auth_ref = identity_fakes.fake_auth_ref(identity_fakes.TOKEN)
        self.ar_mock = mock.PropertyMock(return_value=auth_ref)
        type(self.app.client_manager).auth_ref = self.ar_mock
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        collist = ('expires', 'id', 'project_id', 'user_id')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.token_expires, identity_fakes.token_id, 'project-id', 'user-id')
        self.assertEqual(datalist, data)

    def test_token_issue_with_unscoped_token(self):
        auth_ref = identity_fakes.fake_auth_ref(identity_fakes.UNSCOPED_TOKEN)
        self.ar_mock = mock.PropertyMock(return_value=auth_ref)
        type(self.app.client_manager).auth_ref = self.ar_mock
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        collist = ('expires', 'id', 'user_id')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.token_expires, identity_fakes.token_id, 'user-id')
        self.assertEqual(datalist, data)