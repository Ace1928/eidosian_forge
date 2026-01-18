import copy
from openstackclient.identity.v3 import consumer
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestConsumerSet(TestOAuth1):

    def setUp(self):
        super(TestConsumerSet, self).setUp()
        self.consumers_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.OAUTH_CONSUMER), loaded=True)
        consumer_updated = copy.deepcopy(identity_fakes.OAUTH_CONSUMER)
        consumer_updated['description'] = 'consumer new description'
        self.consumers_mock.update.return_value = fakes.FakeResource(None, consumer_updated, loaded=True)
        self.cmd = consumer.SetConsumer(self.app, None)

    def test_consumer_update(self):
        new_description = 'consumer new description'
        arglist = ['--description', new_description, identity_fakes.consumer_id]
        verifylist = [('description', new_description), ('consumer', identity_fakes.consumer_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'description': new_description}
        self.consumers_mock.update.assert_called_with(identity_fakes.consumer_id, **kwargs)
        self.assertIsNone(result)