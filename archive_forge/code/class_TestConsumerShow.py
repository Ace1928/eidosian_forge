import copy
from openstackclient.identity.v3 import consumer
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestConsumerShow(TestOAuth1):

    def setUp(self):
        super(TestConsumerShow, self).setUp()
        consumer_no_secret = copy.deepcopy(identity_fakes.OAUTH_CONSUMER)
        del consumer_no_secret['secret']
        self.consumers_mock.get.return_value = fakes.FakeResource(None, consumer_no_secret, loaded=True)
        self.cmd = consumer.ShowConsumer(self.app, None)

    def test_consumer_show(self):
        arglist = [identity_fakes.consumer_id]
        verifylist = [('consumer', identity_fakes.consumer_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.consumers_mock.get.assert_called_with(identity_fakes.consumer_id)
        collist = ('description', 'id')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.consumer_description, identity_fakes.consumer_id)
        self.assertEqual(datalist, data)