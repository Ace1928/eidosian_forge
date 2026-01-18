from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as client
from heat.engine import resource
from heat.engine.resources.openstack.mistral import external_resource
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class TestMistralExternalResource(common.HeatTestCase):

    def setUp(self):
        super(TestMistralExternalResource, self).setUp()
        self.ctx = utils.dummy_context()
        tmpl = template_format.parse(external_resource_template)
        self.stack = utils.parse_stack(tmpl, stack_name='test_stack')
        resource_defns = self.stack.t.resource_definitions(self.stack)
        self.rsrc_defn = resource_defns['custom']
        self.mistral = mock.Mock()
        self.patchobject(external_resource.MistralExternalResource, 'client', return_value=self.mistral)
        self.patchobject(client.MistralClientPlugin, '_create')
        self.client = client.MistralClientPlugin(self.ctx)

    def _create_resource(self, name, snippet, stack, output='{}', get_state='SUCCESS'):
        execution = external_resource.MistralExternalResource(name, snippet, stack)
        self.mistral.executions.get.return_value = FakeExecution('test_stack-execution-b5fiekfci3yc', output, get_state)
        self.mistral.executions.create.return_value = FakeExecution('test_stack-execution-b5fiekfci3yc')
        return execution

    def test_create(self):
        execution = self._create_resource('execution', self.rsrc_defn, self.stack)
        scheduler.TaskRunner(execution.create)()
        expected_state = (execution.CREATE, execution.COMPLETE)
        self.assertEqual(expected_state, execution.state)
        self.assertEqual('test_stack-execution-b5fiekfci3yc', execution.resource_id)

    def test_create_with_resource_id_output(self):
        output = '{"resource_id": "my-fake-resource-id"}'
        execution = self._create_resource('execution', self.rsrc_defn, self.stack, output)
        scheduler.TaskRunner(execution.create)()
        expected_state = (execution.CREATE, execution.COMPLETE)
        self.assertEqual(expected_state, execution.state)
        self.assertEqual('my-fake-resource-id', execution.resource_id)

    def test_replace_on_change(self):
        execution = self._create_resource('execution', self.rsrc_defn, self.stack)
        scheduler.TaskRunner(execution.create)()
        expected_state = (execution.CREATE, execution.COMPLETE)
        self.assertEqual(expected_state, execution.state)
        tmpl = template_format.parse(external_resource_template)
        tmpl['resources']['custom']['properties']['input']['foo2'] = '4567'
        res_defns = template.Template(tmpl).resource_definitions(self.stack)
        new_custom_defn = res_defns['custom']
        self.assertRaises(resource.UpdateReplace, scheduler.TaskRunner(execution.update, new_custom_defn))

    def test_create_failed(self):
        execution = self._create_resource('execution', self.rsrc_defn, self.stack, get_state='ERROR')
        self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(execution.create))
        expected_state = (execution.CREATE, execution.FAILED)
        self.assertEqual(expected_state, execution.state)