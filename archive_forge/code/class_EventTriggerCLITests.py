import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
class EventTriggerCLITests(base_v2.MistralClientTestBase):
    """Test suite checks commands to work with event-triggers."""

    @classmethod
    def setUpClass(cls):
        super(EventTriggerCLITests, cls).setUpClass()

    def setUp(self):
        super(EventTriggerCLITests, self).setUp()
        wf = self.workflow_create(self.wf_def)
        self.wf_id = wf[0]['ID']

    def test_event_trigger_create_delete(self):
        trigger = self.mistral_admin('event-trigger-create', params='trigger %s dummy_exchange dummy_topic event.dummy {}' % self.wf_id)
        self.assertTableStruct(trigger, ['Field', 'Value'])
        tr_id = self.get_field_value(trigger, 'ID')
        tr_name = self.get_field_value(trigger, 'Name')
        wf_id = self.get_field_value(trigger, 'Workflow ID')
        created_at = self.get_field_value(trigger, 'Created at')
        self.assertEqual('trigger', tr_name)
        self.assertEqual(self.wf_id, wf_id)
        self.assertIsNotNone(created_at)
        triggers = self.mistral_admin('event-trigger-list')
        self.assertIn(tr_name, [tr['Name'] for tr in triggers])
        self.assertIn(wf_id, [tr['Workflow ID'] for tr in triggers])
        self.mistral('event-trigger-delete', params=tr_id)
        triggers = self.mistral_admin('event-trigger-list')
        self.assertNotIn(tr_name, [tr['Name'] for tr in triggers])

    def test_two_event_triggers_for_one_wf(self):
        self.event_trigger_create('trigger1', self.wf_id, 'dummy_exchange', 'dummy_topic', 'event.dummy', '{}')
        self.event_trigger_create('trigger2', self.wf_id, 'dummy_exchange', 'dummy_topic', 'dummy.event', '{}')
        triggers = self.mistral_admin('event-trigger-list')
        self.assertIn('trigger1', [tr['Name'] for tr in triggers])
        self.assertIn('trigger2', [tr['Name'] for tr in triggers])

    def test_event_trigger_get(self):
        trigger = self.event_trigger_create('trigger', self.wf_id, 'dummy_exchange', 'dummy_topic', 'event.dummy.other', '{}')
        self.assertTableStruct(trigger, ['Field', 'Value'])
        ev_tr_id = self.get_field_value(trigger, 'ID')
        fetched_tr = self.mistral_admin('event-trigger-get', params=ev_tr_id)
        self.assertTableStruct(trigger, ['Field', 'Value'])
        tr_name = self.get_field_value(fetched_tr, 'Name')
        wf_id = self.get_field_value(fetched_tr, 'Workflow ID')
        created_at = self.get_field_value(fetched_tr, 'Created at')
        self.assertEqual('trigger', tr_name)
        self.assertEqual(self.wf_id, wf_id)
        self.assertIsNotNone(created_at)