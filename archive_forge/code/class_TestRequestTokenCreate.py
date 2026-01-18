import copy
from openstackclient.identity.v3 import token
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRequestTokenCreate(TestOAuth1):

    def setUp(self):
        super(TestRequestTokenCreate, self).setUp()
        self.request_tokens_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.OAUTH_REQUEST_TOKEN), loaded=True)
        self.projects_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.PROJECT), loaded=True)
        self.cmd = token.CreateRequestToken(self.app, None)

    def test_create_request_tokens(self):
        arglist = ['--consumer-key', identity_fakes.consumer_id, '--consumer-secret', identity_fakes.consumer_secret, '--project', identity_fakes.project_id]
        verifylist = [('consumer_key', identity_fakes.consumer_id), ('consumer_secret', identity_fakes.consumer_secret), ('project', identity_fakes.project_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.request_tokens_mock.create.assert_called_with(identity_fakes.consumer_id, identity_fakes.consumer_secret, identity_fakes.project_id)
        collist = ('expires', 'id', 'key', 'secret')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.request_token_expires, identity_fakes.request_token_id, identity_fakes.request_token_id, identity_fakes.request_token_secret)
        self.assertEqual(datalist, data)