import copy
from openstackclient.identity.v3 import consumer
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestConsumerCreate(TestOAuth1):

    def setUp(self):
        super(TestConsumerCreate, self).setUp()
        self.consumers_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.OAUTH_CONSUMER), loaded=True)
        self.cmd = consumer.CreateConsumer(self.app, None)

    def test_create_consumer(self):
        arglist = ['--description', identity_fakes.consumer_description]
        verifylist = [('description', identity_fakes.consumer_description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.consumers_mock.create.assert_called_with(identity_fakes.consumer_description)
        collist = ('description', 'id', 'secret')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.consumer_description, identity_fakes.consumer_id, identity_fakes.consumer_secret)
        self.assertEqual(datalist, data)