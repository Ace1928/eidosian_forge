from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2.executions import Execution
from mistralclient.api.v2 import tasks
from mistralclient.commands.v2 import tasks as task_cmd
from mistralclient.tests.unit import base
class TestCLITasksV2(base.BaseCommandTest):

    def test_list(self):
        self.client.tasks.list.return_value = [TASK]
        result = self.call(task_cmd.List)
        self.assertEqual([EXPECTED_TASK_RESULT], result[1])
        self.assertEqual(self.client.tasks.list.call_args[1]['fields'], task_cmd.TaskFormatter.fields())

    def test_list_with_workflow_execution(self):
        self.client.tasks.list.return_value = [TASK]
        result = self.call(task_cmd.List, app_args=['workflow_execution'])
        self.assertEqual([EXPECTED_TASK_RESULT], result[1])

    def test_get(self):
        self.client.tasks.get.return_value = TASK
        result = self.call(task_cmd.Get, app_args=['id'])
        self.assertEqual(EXPECTED_TASK_RESULT, result[1])

    def test_get_result(self):
        self.client.tasks.get.return_value = TASK_WITH_RESULT
        self.call(task_cmd.GetResult, app_args=['id'])
        self.assertDictEqual(TASK_RESULT, jsonutils.loads(self.app.stdout.write.call_args[0][0]))

    def test_get_published(self):
        self.client.tasks.get.return_value = TASK_WITH_PUBLISHED
        self.call(task_cmd.GetPublished, app_args=['id'])
        self.assertDictEqual(TASK_PUBLISHED, jsonutils.loads(self.app.stdout.write.call_args[0][0]))

    def test_rerun(self):
        self.client.tasks.rerun.return_value = TASK
        result = self.call(task_cmd.Rerun, app_args=['id'])
        self.assertEqual(EXPECTED_TASK_RESULT, result[1])

    def test_rerun_no_reset(self):
        self.client.tasks.rerun.return_value = TASK
        result = self.call(task_cmd.Rerun, app_args=['id', '--resume'])
        self.assertEqual(EXPECTED_TASK_RESULT, result[1])

    def test_rerun_update_env(self):
        self.client.tasks.rerun.return_value = TASK
        result = self.call(task_cmd.Rerun, app_args=['id', '--env', '{"k1": "foobar"}'])
        self.assertEqual(EXPECTED_TASK_RESULT, result[1])

    def test_rerun_no_reset_update_env(self):
        self.client.tasks.rerun.return_value = TASK
        result = self.call(task_cmd.Rerun, app_args=['id', '--resume', '--env', '{"k1": "foobar"}'])
        self.assertEqual(EXPECTED_TASK_RESULT, result[1])

    def test_sub_executions(self):
        self.client.tasks.get_task_sub_executions.return_value = TASK_SUB_WF_EXEC
        result = self.call(task_cmd.SubExecutionsLister, app_args=[TASK_DICT['id']])
        self.assertEqual([TASK_SUB_WF_EX_RESULT], result[1])
        self.assertEqual(1, self.client.tasks.get_task_sub_executions.call_count)
        self.assertEqual([mock.call(TASK_DICT['id'], errors_only='', max_depth=-1)], self.client.tasks.get_task_sub_executions.call_args_list)

    def test_sub_executions_errors_only(self):
        self.client.tasks.get_task_sub_executions.return_value = TASK_SUB_WF_EXEC
        self.call(task_cmd.SubExecutionsLister, app_args=[TASK_DICT['id'], '--errors-only'])
        self.assertEqual(1, self.client.tasks.get_task_sub_executions.call_count)
        self.assertEqual([mock.call(TASK_DICT['id'], errors_only=True, max_depth=-1)], self.client.tasks.get_task_sub_executions.call_args_list)

    def test_sub_executions_with_max_depth(self):
        self.client.tasks.get_task_sub_executions.return_value = TASK_SUB_WF_EXEC
        self.call(task_cmd.SubExecutionsLister, app_args=[TASK_DICT['id'], '--max-depth', '3'])
        self.assertEqual(1, self.client.tasks.get_task_sub_executions.call_count)
        self.assertEqual([mock.call(TASK_DICT['id'], errors_only='', max_depth=3)], self.client.tasks.get_task_sub_executions.call_args_list)