from unittest import mock
from urllib import parse as urlparse
from heat.common import template_format
from heat.engine.clients import client_plugin
from heat.engine import resource
from heat.engine.resources.openstack.zaqar import queue
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class ZaqarMessageQueueTest(common.HeatTestCase):

    def setUp(self):
        super(ZaqarMessageQueueTest, self).setUp()
        self.fc = FakeClient()
        self.patchobject(resource.Resource, 'client', return_value=self.fc)
        self.ctx = utils.dummy_context()

    def parse_stack(self, t):
        stack_name = 'test_stack'
        tmpl = template.Template(t)
        self.stack = stack.Stack(self.ctx, stack_name, tmpl)
        self.stack.validate()
        self.stack.store()

    def test_create(self):
        t = template_format.parse(wp_template)
        self.parse_stack(t)
        queue = self.stack['MyQueue2']
        queue_metadata = queue.properties.get('metadata')
        fake_q = FakeQueue(queue.physical_resource_name(), auto_create=False)
        fake_q.metadata.return_value = queue_metadata
        self.fc.queue.return_value = fake_q
        scheduler.TaskRunner(queue.create)()
        self.assertEqual('http://127.0.0.1:8888/v1.1/queues/myqueue', queue.FnGetAtt('href'))
        self.fc.queue.assert_called_once_with(queue.physical_resource_name(), auto_create=False)
        fake_q.metadata.assert_called_once_with(new_meta=queue_metadata)

    def test_create_default_name(self):
        t = template_format.parse(wp_template)
        del t['Resources']['MyQueue2']['Properties']['name']
        self.parse_stack(t)
        queue = self.stack['MyQueue2']
        name_match = utils.PhysName(self.stack.name, 'MyQueue2')
        self.fc.queue.side_effect = FakeQueue
        scheduler.TaskRunner(queue.create)()
        queue_name = queue.physical_resource_name()
        self.assertEqual(name_match, queue_name)
        self.fc.api_version = 2
        self.assertEqual('http://127.0.0.1:8888/v2/queues/' + queue_name, queue.FnGetAtt('href'))
        self.fc.queue.assert_called_once_with(name_match, auto_create=False)

    def test_delete(self):
        t = template_format.parse(wp_template)
        self.parse_stack(t)
        queue = self.stack['MyQueue2']
        queue.resource_id_set(queue.properties.get('name'))
        fake_q = FakeQueue('myqueue', auto_create=False)
        self.fc.queue.return_value = fake_q
        scheduler.TaskRunner(queue.create)()
        self.fc.queue.assert_called_once_with('myqueue', auto_create=False)
        scheduler.TaskRunner(queue.delete)()
        fake_q.delete.assert_called()

    @mock.patch.object(queue.ZaqarQueue, 'client')
    def test_delete_not_found(self, mockclient):

        class ZaqarClientPlugin(client_plugin.ClientPlugin):

            def _create(self):
                return mockclient()
        mock_def = mock.Mock(spec=rsrc_defn.ResourceDefinition)
        mock_def.resource_type = 'OS::Zaqar::Queue'
        props = mock.Mock()
        props.props = {}
        mock_def.properties.return_value = props
        stack = utils.parse_stack(template_format.parse(wp_template))
        self.patchobject(stack, 'db_resource_get', return_value=None)
        mockplugin = ZaqarClientPlugin(self.ctx)
        clients = self.patchobject(stack, 'clients')
        clients.client_plugin.return_value = mockplugin
        mockplugin.is_not_found = mock.Mock()
        mockplugin.is_not_found.return_value = True
        zaqar_q = mock.Mock()
        zaqar_q.delete.side_effect = ResourceNotFound()
        mockclient.return_value.queue.return_value = zaqar_q
        zplugin = queue.ZaqarQueue('test_delete_not_found', mock_def, stack)
        zplugin.resource_id = 'test_delete_not_found'
        zplugin.handle_delete()
        clients.client_plugin.assert_called_once_with('zaqar')
        mockplugin.is_not_found.assert_called_once_with(zaqar_q.delete.side_effect)
        mockclient.return_value.queue.assert_called_once_with('test_delete_not_found', auto_create=False)

    def test_update_in_place(self):
        t = template_format.parse(wp_template)
        self.parse_stack(t)
        queue = self.stack['MyQueue2']
        queue.resource_id_set(queue.properties.get('name'))
        fake_q = FakeQueue('myqueue', auto_create=False)
        self.fc.queue.return_value = fake_q
        t = template_format.parse(wp_template)
        new_queue = t['Resources']['MyQueue2']
        new_queue['Properties']['metadata'] = {'key1': 'value'}
        resource_defns = template.Template(t).resource_definitions(self.stack)
        scheduler.TaskRunner(queue.create)()
        self.fc.queue.assert_called_once_with('myqueue', auto_create=False)
        fake_q.metadata.assert_called_with(new_meta={'key1': {'key2': 'value', 'key3': [1, 2]}})
        scheduler.TaskRunner(queue.update, resource_defns['MyQueue2'])()
        fake_q.metadata.assert_called_with(new_meta={'key1': 'value'})

    def test_update_replace(self):
        t = template_format.parse(wp_template)
        self.parse_stack(t)
        queue = self.stack['MyQueue2']
        queue.resource_id_set(queue.properties.get('name'))
        fake_q = FakeQueue('myqueue', auto_create=False)
        self.fc.queue.return_value = fake_q
        t = template_format.parse(wp_template)
        t['Resources']['MyQueue2']['Properties']['name'] = 'new_queue'
        resource_defns = template.Template(t).resource_definitions(self.stack)
        new_queue = resource_defns['MyQueue2']
        scheduler.TaskRunner(queue.create)()
        self.fc.queue.assert_called_once_with('myqueue', auto_create=False)
        err = self.assertRaises(resource.UpdateReplace, scheduler.TaskRunner(queue.update, new_queue))
        msg = 'The Resource MyQueue2 requires replacement.'
        self.assertEqual(msg, str(err))

    def test_show_resource(self):
        t = template_format.parse(wp_template)
        self.parse_stack(t)
        queue = self.stack['MyQueue2']
        fake_q = FakeQueue(queue.physical_resource_name(), auto_create=False)
        queue_metadata = queue.properties.get('metadata')
        fake_q.metadata.return_value = queue_metadata
        self.fc.queue.return_value = fake_q
        scheduler.TaskRunner(queue.create)()
        self.fc.queue.assert_called_once_with(queue.physical_resource_name(), auto_create=False)
        self.assertEqual({'metadata': {'key1': {'key2': 'value', 'key3': [1, 2]}}}, queue._show_resource())
        fake_q.metadata.assert_called_with()

    def test_parse_live_resource_data(self):
        t = template_format.parse(wp_template)
        self.parse_stack(t)
        queue = self.stack['MyQueue2']
        fake_q = FakeQueue(queue.physical_resource_name(), auto_create=False)
        self.fc.queue.return_value = fake_q
        queue_metadata = queue.properties.get('metadata')
        fake_q.metadata.return_value = queue_metadata
        scheduler.TaskRunner(queue.create)()
        fake_q.metadata.assert_called_with(new_meta=queue_metadata)
        self.fc.queue.assert_called_once_with(queue.physical_resource_name(), auto_create=False)
        self.assertEqual({'metadata': {'key1': {'key2': 'value', 'key3': [1, 2]}}, 'name': queue.resource_id}, queue.parse_live_resource_data(queue.properties, queue._show_resource()))
        fake_q.metadata.assert_called_with()