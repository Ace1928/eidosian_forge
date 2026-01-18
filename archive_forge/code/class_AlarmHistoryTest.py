from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from aodhclient.tests.functional import base
class AlarmHistoryTest(base.ClientTestBase):

    def test_help(self):
        self.aodh('help', params='alarm-history show')
        self.aodh('help', params='alarm-history search')

    def test_alarm_history_scenario(self):
        PROJECT_ID = uuidutils.generate_uuid()
        RESOURCE_ID = uuidutils.generate_uuid()
        result = self.aodh(u'alarm', params=u'create --type gnocchi_resources_threshold --name history1 --metric cpu_util --threshold 5 --resource-id %s --resource-type generic --aggregation-method last --project-id %s' % (RESOURCE_ID, PROJECT_ID))
        alarm = self.details_multiple(result)[0]
        ALARM_ID = alarm['alarm_id']
        result = self.aodh(u'alarm', params=u'create --type gnocchi_resources_threshold --name history2 --metric cpu_util --threshold 10 --resource-id %s --resource-type generic --aggregation-method last --project-id %s' % (RESOURCE_ID, PROJECT_ID))
        alarm = self.details_multiple(result)[0]
        ALARM_ID2 = alarm['alarm_id']
        result = self.aodh('alarm-history', params='show %s --limit 1' % ALARM_ID)
        alarm_list = self.parser.listing(result)
        self.assertEqual(1, len(alarm_list))
        result = self.aodh('alarm-history', params='show %s --sort timestamp:asc' % ALARM_ID)
        alarm_history_list = self.parser.listing(result)
        timestamp = [r['timestamp'] for r in alarm_history_list]
        sorted_timestamp = sorted(timestamp)
        self.assertEqual(sorted_timestamp, timestamp)
        result = self.aodh('alarm-history', params='show %s --sort type:desc --sort timestamp:asc' % ALARM_ID)
        alarm_history_list = self.parser.listing(result)
        creation = alarm_history_list.pop(-1)
        timestamp = [r['timestamp'] for r in alarm_history_list]
        sorted_timestamp = sorted(timestamp)
        self.assertEqual(sorted_timestamp, timestamp)
        self.assertEqual('creation', creation['type'])
        result = self.aodh('alarm-history', params='show %s' % ALARM_ID)
        history = self.parser.listing(result)[0]
        for key in ['timestamp', 'type', 'detail', 'event_id']:
            self.assertIn(key, history)
        result = self.aodh('alarm-history', params='show %s' % ALARM_ID)
        history = self.parser.listing(result)[0]
        self.assertEqual('creation', history['type'])
        self.assertEqual('history1', jsonutils.loads(history['detail'])['name'])
        result = self.aodh('alarm-history', params='show %s' % ALARM_ID2)
        history = self.parser.listing(result)[0]
        self.assertEqual('creation', history['type'])
        self.assertEqual('history2', jsonutils.loads(history['detail'])['name'])
        result = self.aodh('alarm-history', params='search')
        self.assertIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])
        self.assertIn(ALARM_ID2, [r['alarm_id'] for r in self.parser.listing(result)])
        result = self.aodh('alarm-history', params='search --query alarm_id=%s' % ALARM_ID)
        history = self.parser.listing(result)[0]
        self.assertEqual(ALARM_ID, history['alarm_id'])
        self.assertEqual('creation', history['type'])
        self.assertEqual('history1', jsonutils.loads(history['detail'])['name'])
        self.aodh('alarm', params='delete %s' % ALARM_ID)
        self.aodh('alarm', params='delete %s' % ALARM_ID2)