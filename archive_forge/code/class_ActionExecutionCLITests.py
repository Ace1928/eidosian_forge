import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
class ActionExecutionCLITests(base_v2.MistralClientTestBase):
    """Test suite checks commands to work with action executions."""

    def setUp(self):
        super(ActionExecutionCLITests, self).setUp()
        wfs = self.workflow_create(self.wf_def)
        self.direct_wf = wfs[0]
        direct_wf_exec = self.execution_create(self.direct_wf['Name'])
        self.direct_ex_id = self.get_field_value(direct_wf_exec, 'ID')

    def test_act_execution_get(self):
        self.wait_execution_success(self.direct_ex_id)
        task = self.mistral_admin('task-list', params=self.direct_ex_id)[0]
        act_ex_from_list = self.mistral_admin('action-execution-list', params=task['ID'])[0]
        act_ex = self.mistral_admin('action-execution-get', params=act_ex_from_list['ID'])
        wf_name = self.get_field_value(act_ex, 'Workflow name')
        state = self.get_field_value(act_ex, 'State')
        self.assertEqual(act_ex_from_list['ID'], self.get_field_value(act_ex, 'ID'))
        self.assertEqual(self.direct_wf['Name'], wf_name)
        self.assertEqual('SUCCESS', state)

    def test_act_execution_list_with_limit(self):
        self.wait_execution_success(self.direct_ex_id)
        act_execs = self.mistral_admin('action-execution-list')
        self.assertGreater(len(act_execs), 1)
        act_execs = self.mistral_admin('action-execution-list', params='--limit 1')
        self.assertEqual(len(act_execs), 1)
        act_ex = act_execs[0]
        self.assertEqual(self.direct_wf['Name'], act_ex['Workflow name'])
        self.assertEqual('SUCCESS', act_ex['State'])

    def test_act_execution_get_list_within_namespace(self):
        namespace = 'bbb'
        self.workflow_create(self.wf_def, namespace=namespace)
        wf_ex = self.execution_create(self.direct_wf['Name'] + ' --namespace ' + namespace)
        exec_id = self.get_field_value(wf_ex, 'ID')
        self.wait_execution_success(exec_id)
        task = self.mistral_admin('task-list', params=exec_id)[0]
        act_ex_from_list = self.mistral_admin('action-execution-list', params=task['ID'])[0]
        act_ex = self.mistral_admin('action-execution-get', params=act_ex_from_list['ID'])
        wf_name = self.get_field_value(act_ex, 'Workflow name')
        wf_namespace = self.get_field_value(act_ex, 'Workflow namespace')
        status = self.get_field_value(act_ex, 'State')
        self.assertEqual(act_ex_from_list['ID'], self.get_field_value(act_ex, 'ID'))
        self.assertEqual(self.direct_wf['Name'], wf_name)
        self.assertEqual('SUCCESS', status)
        self.assertEqual(namespace, wf_namespace)
        self.assertEqual(namespace, act_ex_from_list['Workflow namespace'])

    def test_act_execution_create_delete(self):
        action_ex = self.mistral_admin('run-action', params="std.echo '{0}' --save-result".format('{"output": "Hello!"}'))
        action_ex_id = self.get_field_value(action_ex, 'ID')
        self.assertTableStruct(action_ex, ['Field', 'Value'])
        name = self.get_field_value(action_ex, 'Name')
        wf_name = self.get_field_value(action_ex, 'Workflow name')
        task_name = self.get_field_value(action_ex, 'Task name')
        self.assertEqual('std.echo', name)
        self.assertEqual('None', wf_name)
        self.assertEqual('None', task_name)
        action_exs = self.mistral_admin('action-execution-list')
        self.assertIn(action_ex_id, [ex['ID'] for ex in action_exs])
        self.mistral_admin('action-execution-delete', params=action_ex_id)
        action_exs = self.mistral_admin('action-execution-list')
        self.assertNotIn(action_ex_id, [ex['ID'] for ex in action_exs])