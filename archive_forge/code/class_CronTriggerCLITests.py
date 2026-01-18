import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
class CronTriggerCLITests(base_v2.MistralClientTestBase):
    """Test suite checks commands to work with cron-triggers."""

    @classmethod
    def setUpClass(cls):
        super(CronTriggerCLITests, cls).setUpClass()

    def setUp(self):
        super(CronTriggerCLITests, self).setUp()
        wf = self.workflow_create(self.wf_def)
        self.wf_name = wf[0]['Name']

    def test_cron_trigger_create_delete(self):
        trigger = self.mistral_admin('cron-trigger-create', params='trigger %s {} --pattern "5 * * * *" --count 5 --first-time "4242-12-25 13:37" --utc' % self.wf_name)
        self.assertTableStruct(trigger, ['Field', 'Value'])
        tr_name = self.get_field_value(trigger, 'Name')
        wf_name = self.get_field_value(trigger, 'Workflow')
        created_at = self.get_field_value(trigger, 'Created at')
        remain = self.get_field_value(trigger, 'Remaining executions')
        next_time = self.get_field_value(trigger, 'Next execution time')
        self.assertEqual('trigger', tr_name)
        self.assertEqual(self.wf_name, wf_name)
        self.assertIsNotNone(created_at)
        self.assertEqual('4242-12-25 13:37:00', next_time)
        self.assertEqual('5', remain)
        triggers = self.mistral_admin('cron-trigger-list')
        self.assertIn(tr_name, [tr['Name'] for tr in triggers])
        self.assertIn(wf_name, [tr['Workflow'] for tr in triggers])
        self.mistral('cron-trigger-delete', params=tr_name)
        triggers = self.mistral_admin('cron-trigger-list')
        self.assertNotIn(tr_name, [tr['Name'] for tr in triggers])

    def test_two_cron_triggers_for_one_wf(self):
        self.cron_trigger_create('trigger1', self.wf_name, '{}', '5 * * * *')
        self.cron_trigger_create('trigger2', self.wf_name, '{}', '15 * * * *')
        triggers = self.mistral_admin('cron-trigger-list')
        self.assertIn('trigger1', [tr['Name'] for tr in triggers])
        self.assertIn('trigger2', [tr['Name'] for tr in triggers])

    def test_cron_trigger_get(self):
        trigger = self.cron_trigger_create('trigger', self.wf_name, '{}', '5 * * * *')
        self.assertTableStruct(trigger, ['Field', 'Value'])
        fetched_tr = self.mistral_admin('cron-trigger-get', params='trigger')
        self.assertTableStruct(trigger, ['Field', 'Value'])
        tr_name = self.get_field_value(fetched_tr, 'Name')
        wf_name = self.get_field_value(fetched_tr, 'Workflow')
        created_at = self.get_field_value(fetched_tr, 'Created at')
        self.assertEqual('trigger', tr_name)
        self.assertEqual(self.wf_name, wf_name)
        self.assertIsNotNone(created_at)