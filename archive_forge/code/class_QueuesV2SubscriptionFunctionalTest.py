import json
from unittest import mock
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
class QueuesV2SubscriptionFunctionalTest(base.QueuesTestBase):

    def setUp(self):
        super(QueuesV2SubscriptionFunctionalTest, self).setUp()
        self.queue_name = 'beijing'
        queue = self.client.queue(self.queue_name, force_create=True)
        self.addCleanup(queue.delete)
        subscription_data_1 = {'subscriber': 'http://trigger.me', 'ttl': 3600}
        subscription_data_2 = {'subscriber': 'http://trigger.he', 'ttl': 7200, 'options': {'check everything': True}}
        self.subscription_1 = self.client.subscription(self.queue_name, **subscription_data_1)
        self.addCleanup(self.subscription_1.delete)
        self.subscription_2 = self.client.subscription(self.queue_name, **subscription_data_2)
        self.addCleanup(self.subscription_2.delete)

    def test_subscription_create(self):
        self.assertEqual('http://trigger.me', self.subscription_1.subscriber)
        self.assertEqual(3600, self.subscription_1.ttl)
        self.assertEqual('http://trigger.he', self.subscription_2.subscriber)
        self.assertEqual(7200, self.subscription_2.ttl)

    def test_subscription_create_duplicate(self):
        subscription_data_2 = {'subscriber': 'http://trigger.he', 'ttl': 7200, 'options': {'check everything': True}}
        new_subscription = self.client.subscription('beijing', **subscription_data_2)
        self.assertEqual(new_subscription.id, self.subscription_2.id)

    def test_subscription_update(self):
        sub = self.client.subscription(self.queue_name, auto_create=False, **{'id': self.subscription_1.id})
        data = {'subscriber': 'http://trigger.ok', 'ttl': 1000}
        sub.update(data)
        self.assertEqual('http://trigger.ok', sub.subscriber)
        self.assertEqual(1000, sub.ttl)

    def test_subscription_delete(self):
        self.subscription_1.delete()
        subscription_data = {'id': self.subscription_1.id}
        self.assertRaises(errors.ResourceNotFound, self.client.subscription, self.queue_name, **subscription_data)

    def test_subscription_get(self):
        kwargs = {'id': self.subscription_1.id}
        subscription_get = self.client.subscription(self.queue_name, **kwargs)
        self.assertEqual('http://trigger.me', subscription_get.subscriber)
        self.assertEqual(3600, subscription_get.ttl)

    def test_subscription_list(self):
        subscriptions = self.client.subscriptions(self.queue_name)
        subscriptions = list(subscriptions)
        self.assertEqual(2, len(subscriptions))
        subscriber_list = [s.subscriber for s in subscriptions]
        self.assertIn('http://trigger.me', subscriber_list)
        self.assertIn('http://trigger.he', subscriber_list)
        for sub in subscriptions:
            if sub.subscriber == 'http://trigger.he':
                self.assertEqual('beijing', sub.queue_name)
                self.assertEqual(self.subscription_2.id, sub.id)
                self.assertEqual(7200, sub.ttl)
                self.assertEqual({'check everything': True}, sub.options)