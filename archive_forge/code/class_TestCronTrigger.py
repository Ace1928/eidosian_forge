from openstack.tests.unit import base
from openstack.workflow.v2 import cron_trigger
class TestCronTrigger(base.TestCase):

    def test_basic(self):
        sot = cron_trigger.CronTrigger()
        self.assertEqual('cron_trigger', sot.resource_key)
        self.assertEqual('cron_triggers', sot.resources_key)
        self.assertEqual('/cron_triggers', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_delete)
        self.assertDictEqual({'marker': 'marker', 'limit': 'limit', 'sort_keys': 'sort_keys', 'sort_dirs': 'sort_dirs', 'fields': 'fields', 'name': 'name', 'workflow_name': 'workflow_name', 'workflow_id': 'workflow_id', 'workflow_input': 'workflow_input', 'workflow_params': 'workflow_params', 'scope': 'scope', 'pattern': 'pattern', 'remaining_executions': 'remaining_executions', 'project_id': 'project_id', 'first_execution_time': 'first_execution_time', 'next_execution_time': 'next_execution_time', 'created_at': 'created_at', 'updated_at': 'updated_at', 'all_projects': 'all_projects'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = cron_trigger.CronTrigger(**FAKE)
        self.assertEqual(FAKE['id'], sot.id)
        self.assertEqual(FAKE['pattern'], sot.pattern)
        self.assertEqual(FAKE['remaining_executions'], sot.remaining_executions)
        self.assertEqual(FAKE['first_execution_time'], sot.first_execution_time)
        self.assertEqual(FAKE['next_execution_time'], sot.next_execution_time)
        self.assertEqual(FAKE['workflow_name'], sot.workflow_name)
        self.assertEqual(FAKE['workflow_id'], sot.workflow_id)
        self.assertEqual(FAKE['workflow_input'], sot.workflow_input)
        self.assertEqual(FAKE['workflow_params'], sot.workflow_params)