import copy
import io
import sys
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import action_executions as action_ex
from mistralclient.commands.v2 import action_executions as action_ex_cmd
from mistralclient.tests.unit import base
class TestCLIActionExecutions(base.BaseCommandTest):

    def test_create(self):
        self.client.action_executions.create.return_value = ACTION_EX_WITH_OUTPUT
        self.call(action_ex_cmd.Create, app_args=['some', '{"output": "Hello!"}'])
        self.assertDictEqual(ACTION_EX_RESULT, jsonutils.loads(self.app.stdout.write.call_args[0][0]))

    def test_create_save_result(self):
        self.client.action_executions.create.return_value = ACTION_EX_WITH_OUTPUT
        result = self.call(action_ex_cmd.Create, app_args=['some', '{"output": "Hello!"}', '--save-result'])
        self.assertEqual(('123', 'some', 'thing', '', 'task1', '1-2-3-4', 'RUNNING', 'RUNNING somehow.', True, '1', '1'), result[1])

    def test_create_run_sync(self):
        self.client.action_executions.create.return_value = ACTION_EX_WITH_OUTPUT
        self.call(action_ex_cmd.Create, app_args=['some', '{"output": "Hello!"}', '--run-sync'])
        self.assertDictEqual(ACTION_EX_RESULT, jsonutils.loads(self.app.stdout.write.call_args[0][0]))

    def test_create_run_sync_and_save_result(self):
        self.client.action_executions.create.return_value = ACTION_EX_WITH_OUTPUT
        self.call(action_ex_cmd.Create, app_args=['some', '{"output": "Hello!"}', '--save-result', '--run-sync'])
        self.assertDictEqual(ACTION_EX_RESULT, jsonutils.loads(self.app.stdout.write.call_args[0][0]))

    def test_update(self):
        states = ['PAUSED', 'RUNNING', 'SUCCESS', 'ERROR', 'CANCELLED']
        for state in states:
            action_ex_dict = copy.deepcopy(ACTION_EX_DICT)
            action_ex_dict['state'] = state
            action_ex_dict['state_info'] = 'testing update'
            action_ex_obj = action_ex.ActionExecution(mock, action_ex_dict)
            self.client.action_executions.update.return_value = action_ex_obj
            result = self.call(action_ex_cmd.Update, app_args=['id', '--state', state])
            expected_result = (action_ex_dict['id'], action_ex_dict['name'], action_ex_dict['workflow_name'], action_ex_dict['workflow_namespace'], action_ex_dict['task_name'], action_ex_dict['task_execution_id'], action_ex_dict['state'], action_ex_dict['state_info'], action_ex_dict['accepted'], action_ex_dict['created_at'], action_ex_dict['updated_at'])
            self.assertEqual(expected_result, result[1])

    def test_update_invalid_state(self):
        states = ['IDLE', 'WAITING', 'DELAYED']
        _stderr = sys.stderr
        sys.stderr = io.StringIO()
        for state in states:
            self.assertRaises(SystemExit, self.call, action_ex_cmd.Update, app_args=['id', '--state', state])
        print(sys.stderr.getvalue())
        sys.stderr = _stderr

    def test_list(self):
        self.client.action_executions.list.return_value = [ACTION_EX]
        result = self.call(action_ex_cmd.List)
        self.assertEqual([('123', 'some', 'thing', '', 'task1', '1-2-3-4', 'RUNNING', True, '1', '1')], result[1])

    def test_get(self):
        self.client.action_executions.get.return_value = ACTION_EX
        result = self.call(action_ex_cmd.Get, app_args=['id'])
        self.assertEqual(('123', 'some', 'thing', '', 'task1', '1-2-3-4', 'RUNNING', 'RUNNING somehow.', True, '1', '1'), result[1])

    def test_get_output(self):
        self.client.action_executions.get.return_value = ACTION_EX_WITH_OUTPUT
        self.call(action_ex_cmd.GetOutput, app_args=['id'])
        self.assertDictEqual(ACTION_EX_RESULT, jsonutils.loads(self.app.stdout.write.call_args[0][0]))

    def test_get_input(self):
        self.client.action_executions.get.return_value = ACTION_EX_WITH_INPUT
        self.call(action_ex_cmd.GetInput, app_args=['id'])
        self.assertDictEqual(ACTION_EX_INPUT, jsonutils.loads(self.app.stdout.write.call_args[0][0]))

    def test_delete(self):
        self.call(action_ex_cmd.Delete, app_args=['id'])
        self.client.action_executions.delete.assert_called_once_with('id')

    def test_delete_with_multi_names(self):
        self.call(action_ex_cmd.Delete, app_args=['id1', 'id2'])
        self.assertEqual(2, self.client.action_executions.delete.call_count)
        self.assertEqual([mock.call('id1'), mock.call('id2')], self.client.action_executions.delete.call_args_list)