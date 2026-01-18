import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
class ExecutionCLITests(base_v2.MistralClientTestBase):
    """Test suite checks commands to work with executions."""

    @classmethod
    def setUpClass(cls):
        super(ExecutionCLITests, cls).setUpClass()

    def setUp(self):
        super(ExecutionCLITests, self).setUp()
        wfs = self.workflow_create(self.wf_def)
        self.async_wf = self.workflow_create(self.async_wf_def)[0]
        self.direct_wf = wfs[0]
        self.reverse_wf = wfs[1]
        self.create_file('input', '{\n    "farewell": "Bye"\n}\n')
        self.create_file('task_name', '{\n    "task_name": "goodbye"\n}\n')

    def test_execution_by_id_of_workflow_within_namespace(self):
        namespace = 'abc'
        wfs = self.workflow_create(self.lowest_level_wf, namespace=namespace)
        wf_def_name = wfs[0]['Name']
        wf_id = wfs[0]['ID']
        execution = self.execution_create(wf_id)
        self.assertTableStruct(execution, ['Field', 'Value'])
        wf_name = self.get_field_value(execution, 'Workflow name')
        wf_namespace = self.get_field_value(execution, 'Workflow namespace')
        wf_id = self.get_field_value(execution, 'Workflow ID')
        self.assertEqual(wf_def_name, wf_name)
        self.assertEqual(namespace, wf_namespace)
        self.assertIsNotNone(wf_id)

    def test_execution_within_namespace_create_delete(self):
        namespace = 'abc'
        self.workflow_create(self.lowest_level_wf)
        self.workflow_create(self.lowest_level_wf, namespace=namespace)
        self.workflow_create(self.middle_wf, namespace=namespace)
        self.workflow_create(self.top_level_wf)
        wfs = self.workflow_create(self.top_level_wf, namespace=namespace)
        top_wf_name = wfs[0]['Name']
        execution = self.mistral_admin('execution-create', params='{0} --namespace {1}'.format(top_wf_name, namespace))
        exec_id = self.get_field_value(execution, 'ID')
        self.assertTableStruct(execution, ['Field', 'Value'])
        wf_name = self.get_field_value(execution, 'Workflow name')
        wf_namespace = self.get_field_value(execution, 'Workflow namespace')
        wf_id = self.get_field_value(execution, 'Workflow ID')
        created_at = self.get_field_value(execution, 'Created at')
        self.assertEqual(top_wf_name, wf_name)
        self.assertEqual(namespace, wf_namespace)
        self.assertIsNotNone(wf_id)
        self.assertIsNotNone(created_at)
        execs = self.mistral_admin('execution-list')
        self.assertIn(exec_id, [ex['ID'] for ex in execs])
        self.assertIn(wf_name, [ex['Workflow name'] for ex in execs])
        self.assertIn(namespace, [ex['Workflow namespace'] for ex in execs])
        params = '{} --force'.format(exec_id)
        self.mistral_admin('execution-delete', params=params)

    def test_execution_create_delete(self):
        execution = self.mistral_admin('execution-create', params='{0} -d "execution test"'.format(self.direct_wf['Name']))
        exec_id = self.get_field_value(execution, 'ID')
        self.assertTableStruct(execution, ['Field', 'Value'])
        wf_name = self.get_field_value(execution, 'Workflow name')
        wf_id = self.get_field_value(execution, 'Workflow ID')
        created_at = self.get_field_value(execution, 'Created at')
        description = self.get_field_value(execution, 'Description')
        self.assertEqual(self.direct_wf['Name'], wf_name)
        self.assertIsNotNone(wf_id)
        self.assertIsNotNone(created_at)
        self.assertEqual('execution test', description)
        execs = self.mistral_admin('execution-list')
        self.assertIn(exec_id, [ex['ID'] for ex in execs])
        self.assertIn(wf_name, [ex['Workflow name'] for ex in execs])
        params = '{} --force'.format(exec_id)
        self.mistral_admin('execution-delete', params=params)

    def test_execution_create_with_input_and_start_task(self):
        execution = self.execution_create('%s input task_name' % self.reverse_wf['Name'])
        exec_id = self.get_field_value(execution, 'ID')
        result = self.wait_execution_success(exec_id)
        self.assertTrue(result)

    def test_execution_update(self):
        execution = self.execution_create(self.async_wf['Name'])
        exec_id = self.get_field_value(execution, 'ID')
        status = self.get_field_value(execution, 'State')
        self.assertEqual('RUNNING', status)
        execution = self.mistral_admin('execution-update', params='{0} -s PAUSED'.format(exec_id))
        updated_exec_id = self.get_field_value(execution, 'ID')
        status = self.get_field_value(execution, 'State')
        self.assertEqual(exec_id, updated_exec_id)
        self.assertEqual('PAUSED', status)
        execution = self.mistral_admin('execution-update', params='{0} -d "execution update test"'.format(exec_id))
        description = self.get_field_value(execution, 'Description')
        self.assertEqual('execution update test', description)

    def test_execution_get(self):
        execution = self.execution_create(self.direct_wf['Name'])
        exec_id = self.get_field_value(execution, 'ID')
        execution = self.mistral_admin('execution-get', params='{0}'.format(exec_id))
        gotten_id = self.get_field_value(execution, 'ID')
        wf_name = self.get_field_value(execution, 'Workflow name')
        wf_id = self.get_field_value(execution, 'Workflow ID')
        self.assertIsNotNone(wf_id)
        self.assertEqual(exec_id, gotten_id)
        self.assertEqual(self.direct_wf['Name'], wf_name)

    def test_execution_get_input(self):
        execution = self.execution_create(self.direct_wf['Name'])
        exec_id = self.get_field_value(execution, 'ID')
        ex_input = self.mistral_admin('execution-get-input', params=exec_id)
        self.assertEqual([], ex_input)

    def test_execution_get_output(self):
        execution = self.execution_create(self.direct_wf['Name'])
        exec_id = self.get_field_value(execution, 'ID')
        ex_output = self.mistral_admin('execution-get-output', params=exec_id)
        self.assertEqual([], ex_output)

    def test_executions_list_with_task(self):
        wrapping_wf = self.workflow_create(self.wf_wrapping_wf)
        decoy = self.execution_create(wrapping_wf[-1]['Name'])
        wrapping_wf_ex = self.execution_create(wrapping_wf[-1]['Name'])
        wrapping_wf_ex_id = self.get_field_value(wrapping_wf_ex, 'ID')
        self.assertIsNot(wrapping_wf_ex_id, self.get_field_value(decoy, 'ID'))
        tasks = self.mistral_admin('task-list', params=wrapping_wf_ex_id)
        wrapping_task_id = tasks[-1]['ID']
        wf_execs = self.mistral_cli(True, 'execution-list', params='--task {}'.format(wrapping_task_id))
        self.assertEqual(1, len(wf_execs))
        wf_exec = wf_execs[0]
        self.assertEqual(wrapping_task_id, wf_exec['Task Execution ID'])

    def test_executions_list_with_pagination(self):
        ex1 = self.execution_create(params='{0} -d "a"'.format(self.direct_wf['Name']))
        time.sleep(1)
        ex2 = self.execution_create(params='{0} -d "b"'.format(self.direct_wf['Name']))
        all_wf_ids = [self.get_field_value(ex1, 'ID'), self.get_field_value(ex2, 'ID')]
        wf_execs = self.mistral_cli(True, 'execution-list')
        self.assertEqual(2, len(wf_execs))
        self.assertEqual(set(all_wf_ids), set([ex['ID'] for ex in wf_execs]))
        wf_execs = self.mistral_cli(True, 'execution-list', params='--oldest --limit 1')
        self.assertEqual(1, len(wf_execs))
        not_expected = wf_execs[0]['ID']
        expected = [ex for ex in all_wf_ids if ex != wf_execs[0]['ID']][0]
        wf_execs = self.mistral_cli(True, 'execution-list', params='--marker %s' % not_expected)
        self.assertNotIn(not_expected, [ex['ID'] for ex in wf_execs])
        self.assertIn(expected, [ex['ID'] for ex in wf_execs])
        wf_execs = self.mistral_cli(True, 'execution-list', params='--sort_keys Description')
        self.assertEqual(set(all_wf_ids), set([ex['ID'] for ex in wf_execs]))
        wf_ex1_index = -1
        wf_ex2_index = -1
        for idx, ex in enumerate(wf_execs):
            if ex['ID'] == all_wf_ids[0]:
                wf_ex1_index = idx
            elif ex['ID'] == all_wf_ids[1]:
                wf_ex2_index = idx
        self.assertLess(wf_ex1_index, wf_ex2_index)
        wf_execs = self.mistral_cli(True, 'execution-list', params='--sort_keys Description --sort_dirs=desc')
        self.assertEqual(set(all_wf_ids), set([ex['ID'] for ex in wf_execs]))
        wf_ex1_index = -1
        wf_ex2_index = -1
        for idx, ex in enumerate(wf_execs):
            if ex['ID'] == all_wf_ids[0]:
                wf_ex1_index = idx
            elif ex['ID'] == all_wf_ids[1]:
                wf_ex2_index = idx
        self.assertGreater(wf_ex1_index, wf_ex2_index)

    def test_execution_list_with_filter(self):
        wf_ex1 = self.execution_create(params='{0} -d "a"'.format(self.direct_wf['Name']))
        wf_ex1_id = self.get_field_value(wf_ex1, 'ID')
        self.execution_create(params='{0} -d "b"'.format(self.direct_wf['Name']))
        wf_execs = self.mistral_cli(True, 'execution-list')
        self.assertTableStruct(wf_execs, ['ID', 'Workflow name', 'Workflow ID', 'State', 'Created at', 'Updated at'])
        self.assertEqual(2, len(wf_execs))
        wf_execs = self.mistral_cli(True, 'execution-list', params='--filter description=a')
        self.assertTableStruct(wf_execs, ['ID', 'Workflow name', 'Workflow ID', 'State', 'Created at', 'Updated at'])
        self.assertEqual(1, len(wf_execs))
        self.assertEqual(wf_ex1_id, wf_execs[0]['ID'])
        self.assertEqual('a', wf_execs[0]['Description'])

    def test_executions_list_with_rootsonly(self):
        wrapping_wf = self.workflow_create(self.wf_wrapping_wf)
        wrapping_wf_ex = self.execution_create(wrapping_wf[-1]['Name'])
        wrapping_wf_ex_id = self.get_field_value(wrapping_wf_ex, 'ID')
        wf_execs = self.mistral_cli(True, 'execution-list', params='--rootsonly')
        self.assertEqual(1, len(wf_execs))
        wf_exec = wf_execs[0]
        self.assertEqual(wrapping_wf_ex_id, wf_exec['ID'])