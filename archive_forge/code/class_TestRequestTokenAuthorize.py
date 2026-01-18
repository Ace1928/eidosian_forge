import copy
from openstackclient.identity.v3 import token
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRequestTokenAuthorize(TestOAuth1):

    def setUp(self):
        super(TestRequestTokenAuthorize, self).setUp()
        self.roles_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE), loaded=True)
        copied_verifier = copy.deepcopy(identity_fakes.OAUTH_VERIFIER)
        resource = fakes.FakeResource(None, copied_verifier, loaded=True)
        self.request_tokens_mock.authorize.return_value = resource
        self.cmd = token.AuthorizeRequestToken(self.app, None)

    def test_authorize_request_tokens(self):
        arglist = ['--request-key', identity_fakes.request_token_id, '--role', identity_fakes.role_name]
        verifylist = [('request_key', identity_fakes.request_token_id), ('role', [identity_fakes.role_name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.request_tokens_mock.authorize.assert_called_with(identity_fakes.request_token_id, [identity_fakes.role_id])
        collist = ('oauth_verifier',)
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.oauth_verifier_pin,)
        self.assertEqual(datalist, data)