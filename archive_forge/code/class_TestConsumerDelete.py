import copy
from openstackclient.identity.v3 import consumer
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestConsumerDelete(TestOAuth1):

    def setUp(self):
        super(TestConsumerDelete, self).setUp()
        self.consumers_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.OAUTH_CONSUMER), loaded=True)
        self.consumers_mock.delete.return_value = None
        self.cmd = consumer.DeleteConsumer(self.app, None)

    def test_delete_consumer(self):
        arglist = [identity_fakes.consumer_id]
        verifylist = [('consumer', [identity_fakes.consumer_id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.consumers_mock.delete.assert_called_with(identity_fakes.consumer_id)
        self.assertIsNone(result)