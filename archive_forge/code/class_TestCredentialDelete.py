from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.identity.v3 import credential
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
class TestCredentialDelete(TestCredential):
    credentials = identity_fakes.FakeCredential.create_credentials(count=2)

    def setUp(self):
        super(TestCredentialDelete, self).setUp()
        self.credentials_mock.delete.return_value = None
        self.cmd = credential.DeleteCredential(self.app, None)

    def test_credential_delete(self):
        arglist = [self.credentials[0].id]
        verifylist = [('credential', [self.credentials[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.credentials_mock.delete.assert_called_with(self.credentials[0].id)
        self.assertIsNone(result)

    def test_credential_multi_delete(self):
        arglist = []
        for c in self.credentials:
            arglist.append(c.id)
        verifylist = [('credential', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for c in self.credentials:
            calls.append(call(c.id))
        self.credentials_mock.delete.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_credential_multi_delete_with_exception(self):
        arglist = [self.credentials[0].id, 'unexist_credential']
        verifylist = [('credential', [self.credentials[0].id, 'unexist_credential'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        delete_mock_result = [None, exceptions.CommandError]
        self.credentials_mock.delete = mock.Mock(side_effect=delete_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 credential failed to delete.', str(e))
        self.credentials_mock.delete.assert_any_call(self.credentials[0].id)
        self.credentials_mock.delete.assert_any_call('unexist_credential')