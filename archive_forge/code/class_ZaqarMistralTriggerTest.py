from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as mistral_client_plugin
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from oslo_serialization import jsonutils
class ZaqarMistralTriggerTest(common.HeatTestCase):

    def setUp(self):
        super(ZaqarMistralTriggerTest, self).setUp()
        self.fc = mock.Mock()
        self.patchobject(resource.Resource, 'client', return_value=self.fc)
        self.ctx = utils.dummy_context()
        self.patchobject(mistral_client_plugin.WorkflowConstraint, 'validate', return_value=True)
        self.subscr_id = '58138648c1e2eb7355d62137'
        stack_name = 'test_stack'
        t = template_format.parse(mistral_template)
        tmpl = template.Template(t)
        self.stack = stack.Stack(self.ctx, stack_name, tmpl)
        self.stack.validate()
        self.stack.store()

        def client(name='zaqar'):
            if name == 'mistral':
                client = mock.Mock()
                http_client = mock.Mock()
                client.executions = mock.Mock(spec=['http_client'])
                client.executions.http_client = http_client
                http_client.base_url = 'http://mistral.example.net:8989'
                return client
            elif name == 'zaqar':
                return self.fc
        self.subscr = self.stack['subscription']
        self.subscr.client = mock.Mock(side_effect=client)
        self.subscriber = 'trust+http://mistral.example.net:8989/executions'
        self.options = {'post_data': JsonString({'workflow_id': 'abcd', 'input': {'key1': 'value1'}, 'params': {'env': {'notification': '$zaqar_message$'}}})}

    def test_create(self):
        subscr = self.subscr
        fake_subscr = FakeSubscription(subscr.properties['queue_name'], self.subscr_id)
        self.fc.subscription.return_value = fake_subscr
        scheduler.TaskRunner(subscr.create)()
        self.assertEqual(self.subscr_id, subscr.FnGetRefId())
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], options=self.options, subscriber=self.subscriber, ttl=220367260800)

    def test_delete(self):
        subscr = self.subscr
        fake_subscr = FakeSubscription(subscr.properties['queue_name'], self.subscr_id)
        self.fc.subscription.return_value = fake_subscr
        scheduler.TaskRunner(subscr.create)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], options=self.options, subscriber=self.subscriber, ttl=220367260800)
        scheduler.TaskRunner(subscr.delete)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], id=self.subscr_id, auto_create=False)
        self.assertTrue(getattr(fake_subscr, '_deleted', False))

    def test_delete_not_found(self):
        subscr = self.subscr
        fake_subscr = FakeSubscription(subscr.properties['queue_name'], self.subscr_id)
        self.fc.subscription.side_effect = [fake_subscr, ResourceNotFound]
        scheduler.TaskRunner(subscr.create)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], options=self.options, subscriber=self.subscriber, ttl=220367260800)
        scheduler.TaskRunner(subscr.delete)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], id=self.subscr_id, auto_create=False)

    def test_update_in_place(self):
        subscr = self.subscr
        fake_subscr = FakeSubscription(subscr.properties['queue_name'], self.subscr_id)
        self.fc.subscription.return_value = fake_subscr
        t = template_format.parse(mistral_template)
        new_subscr = t['resources']['subscription']
        new_subscr['properties']['ttl'] = '3601'
        resource_defns = template.Template(t).resource_definitions(self.stack)
        scheduler.TaskRunner(subscr.create)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], options=self.options, subscriber=self.subscriber, ttl=220367260800)
        scheduler.TaskRunner(subscr.update, resource_defns['subscription'])()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], id=self.subscr_id, auto_create=False)

    def test_update_replace(self):
        subscr = self.subscr
        fake_subscr = FakeSubscription(subscr.properties['queue_name'], self.subscr_id)
        self.fc.subscription.return_value = fake_subscr
        t = template_format.parse(mistral_template)
        t['resources']['subscription']['properties']['queue_name'] = 'foo'
        resource_defns = template.Template(t).resource_definitions(self.stack)
        new_subscr = resource_defns['subscription']
        scheduler.TaskRunner(subscr.create)()
        err = self.assertRaises(resource.UpdateReplace, scheduler.TaskRunner(subscr.update, new_subscr))
        msg = 'The Resource subscription requires replacement.'
        self.assertEqual(msg, str(err))
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], options=self.options, subscriber=self.subscriber, ttl=220367260800)

    def test_show_resource(self):
        subscr = self.subscr
        fake_subscr = FakeSubscription(subscr.properties['queue_name'], self.subscr_id)
        fake_subscr.ttl = 220367260800
        fake_subscr.subscriber = self.subscriber
        fake_subscr.options = {'post_data': str(self.options['post_data'])}
        self.fc.subscription.return_value = fake_subscr
        props = self.stack.t.t['resources']['subscription']['properties']
        scheduler.TaskRunner(subscr.create)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], options=self.options, subscriber=self.subscriber, ttl=220367260800)
        self.assertEqual({'queue_name': props['queue_name'], 'id': self.subscr_id, 'subscriber': self.subscriber, 'options': self.options, 'ttl': 220367260800}, subscr._show_resource())
        self.assertEqual({'queue_name': props['queue_name'], 'workflow_id': props['workflow_id'], 'input': props['input'], 'params': {}, 'ttl': 220367260800}, subscr.parse_live_resource_data(subscr.properties, subscr._show_resource()))
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], id=self.subscr_id, auto_create=False)