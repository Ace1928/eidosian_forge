import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
class AodhClientTest(base.ClientTestBase):

    def test_help(self):
        self.aodh('help', params='alarm create')
        self.aodh('help', params='alarm delete')
        self.aodh('help', params='alarm list')
        self.aodh('help', params='alarm show')
        self.aodh('help', params='alarm update')

    def test_alarm_id_or_name_scenario(self):

        def _test(name):
            params = 'create --type event --name %s' % name
            result = self.aodh('alarm', params=params)
            alarm_id = self.details_multiple(result)[0]['alarm_id']
            params = 'show %s' % name
            result = self.aodh('alarm', params=params)
            self.assertEqual(alarm_id, self.details_multiple(result)[0]['alarm_id'])
            params = 'show %s' % alarm_id
            result = self.aodh('alarm', params=params)
            self.assertEqual(alarm_id, self.details_multiple(result)[0]['alarm_id'])
            params = 'update --state ok %s' % name
            result = self.aodh('alarm', params=params)
            self.assertEqual('ok', self.details_multiple(result)[0]['state'])
            params = 'update --state alarm %s' % alarm_id
            result = self.aodh('alarm', params=params)
            self.assertEqual('alarm', self.details_multiple(result)[0]['state'])
            params = 'update --name another-name %s' % name
            result = self.aodh('alarm', params=params)
            self.assertEqual('another-name', self.details_multiple(result)[0]['name'])
            params = 'update --name %s %s' % (name, alarm_id)
            result = self.aodh('alarm', params=params)
            self.assertEqual(name, self.details_multiple(result)[0]['name'])
            params = 'update --name %s %s' % (name, name)
            result = self.aodh('alarm', params=params)
            self.assertEqual(name, self.details_multiple(result)[0]['name'])
            params = 'update --state ok'
            result = self.aodh('alarm', params=params, fail_ok=True, merge_stderr=True)
            self.assertFirstLineStartsWith(result.splitlines(), 'You need to specify one of alarm ID and alarm name(--name) to update an alarm.')
            params = 'delete %s' % name
            result = self.aodh('alarm', params=params)
            self.assertEqual('', result)
            params = 'create --type event --name %s' % name
            result = self.aodh('alarm', params=params)
            alarm_id = self.details_multiple(result)[0]['alarm_id']
            params = 'delete %s' % alarm_id
            result = self.aodh('alarm', params=params)
            self.assertEqual('', result)
        _test(uuidutils.generate_uuid())
        _test('normal-alarm-name')

    def test_event_scenario(self):
        PROJECT_ID = uuidutils.generate_uuid()
        result = self.aodh(u'alarm', params=u'create --type event --name ev_alarm1 --project-id %s' % PROJECT_ID)
        alarm = self.details_multiple(result)[0]
        ALARM_ID = alarm['alarm_id']
        self.assertEqual('ev_alarm1', alarm['name'])
        self.assertEqual('*', alarm['event_type'])
        result = self.aodh('alarm', params='update %s --severity critical --threshold 10' % ALARM_ID)
        alarm_updated = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
        self.assertEqual('critical', alarm_updated['severity'])
        result = self.aodh('alarm', params='update %s --event-type dummy' % ALARM_ID)
        alarm_updated = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
        self.assertEqual('dummy', alarm_updated['event_type'])
        result = self.aodh('alarm', params='show %s' % ALARM_ID)
        alarm_show = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
        self.assertEqual(PROJECT_ID, alarm_show['project_id'])
        self.assertEqual('ev_alarm1', alarm_show['name'])
        self.assertEqual('dummy', alarm_show['event_type'])
        result = self.aodh('alarm', params='show --name ev_alarm1')
        alarm_show = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
        self.assertEqual(PROJECT_ID, alarm_show['project_id'])
        self.assertEqual('ev_alarm1', alarm_show['name'])
        self.assertEqual('dummy', alarm_show['event_type'])
        self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'show %s --name ev_alarm1' % ALARM_ID)
        result = self.aodh('alarm', params='list --filter all_projects=true')
        self.assertIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])
        output_colums = ['alarm_id', 'type', 'name', 'state', 'severity', 'enabled']
        for alarm_list in self.parser.listing(result):
            self.assertEqual(sorted(output_colums), sorted(alarm_list.keys()))
            if alarm_list['alarm_id'] == ALARM_ID:
                self.assertEqual('ev_alarm1', alarm_list['name'])
        result = self.aodh('alarm', params='list --query project_id=%s' % PROJECT_ID)
        alarm_list = self.parser.listing(result)[0]
        self.assertEqual(ALARM_ID, alarm_list['alarm_id'])
        self.assertEqual('ev_alarm1', alarm_list['name'])
        result = self.aodh('alarm', params='delete %s' % ALARM_ID)
        self.assertEqual('', result)
        result = self.aodh('alarm', params='show %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
        expected = 'Alarm %s not found (HTTP 404)' % ALARM_ID
        self.assertFirstLineStartsWith(result.splitlines(), expected)
        result = self.aodh('alarm', params='delete %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
        self.assertFirstLineStartsWith(result.splitlines(), expected)
        result = self.aodh('alarm', params='list')
        self.assertNotIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])

    def test_composite_scenario(self):
        project_id = uuidutils.generate_uuid()
        res_id = uuidutils.generate_uuid()
        result = self.aodh(u'alarm', params=u'create --type composite --name calarm1 --composite-rule \'{"or":[{"threshold": 0.8, "metric": "cpu_util", "type": "gnocchi_resources_threshold", "resource_type": "generic", "resource_id": "%s", "aggregation_method": "mean"},{"and": [{"threshold": 200, "metric": "disk.iops", "type": "gnocchi_resources_threshold", "resource_type": "generic", "resource_id": "%s", "aggregation_method": "mean"},{"threshold": 1000, "metric": "memory","type": "gnocchi_resources_threshold", "resource_type": "generic", "resource_id": "%s", "aggregation_method": "mean"}]}]}\' --project-id %s' % (res_id, res_id, res_id, project_id))
        alarm = self.details_multiple(result)[0]
        alarm_id = alarm['alarm_id']
        self.assertEqual('calarm1', alarm['name'])
        self.assertEqual('composite', alarm['type'])
        self.assertIn('composite_rule', alarm)
        self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'create --type composite --name calarm1 --project-id %s' % project_id)
        result = self.aodh('alarm', params='update %s --severity critical' % alarm_id)
        alarm_updated = self.details_multiple(result)[0]
        self.assertEqual(alarm_id, alarm_updated['alarm_id'])
        self.assertEqual('critical', alarm_updated['severity'])
        result = self.aodh('alarm', params='show %s' % alarm_id)
        alarm_show = self.details_multiple(result)[0]
        self.assertEqual(alarm_id, alarm_show['alarm_id'])
        self.assertEqual(project_id, alarm_show['project_id'])
        self.assertEqual('calarm1', alarm_show['name'])
        result = self.aodh('alarm', params='show --name calarm1')
        alarm_show = self.details_multiple(result)[0]
        self.assertEqual(alarm_id, alarm_show['alarm_id'])
        self.assertEqual(project_id, alarm_show['project_id'])
        self.assertEqual('calarm1', alarm_show['name'])
        self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'show %s --name calarm1' % alarm_id)
        result = self.aodh('alarm', params='list --filter all_projects=true')
        self.assertIn(alarm_id, [r['alarm_id'] for r in self.parser.listing(result)])
        output_colums = ['alarm_id', 'type', 'name', 'state', 'severity', 'enabled']
        for alarm_list in self.parser.listing(result):
            self.assertEqual(sorted(output_colums), sorted(alarm_list.keys()))
            if alarm_list['alarm_id'] == alarm_id:
                self.assertEqual('calarm1', alarm_list['name'])
        result = self.aodh('alarm', params='list --query project_id=%s' % project_id)
        alarm_list = self.parser.listing(result)[0]
        self.assertEqual(alarm_id, alarm_list['alarm_id'])
        self.assertEqual('calarm1', alarm_list['name'])
        result = self.aodh('alarm', params='delete %s' % alarm_id)
        self.assertEqual('', result)
        result = self.aodh('alarm', params='show %s' % alarm_id, fail_ok=True, merge_stderr=True)
        expected = 'Alarm %s not found (HTTP 404)' % alarm_id
        self.assertFirstLineStartsWith(result.splitlines(), expected)
        result = self.aodh('alarm', params='delete %s' % alarm_id, fail_ok=True, merge_stderr=True)
        self.assertFirstLineStartsWith(result.splitlines(), expected)
        result = self.aodh('alarm', params='list')
        self.assertNotIn(alarm_id, [r['alarm_id'] for r in self.parser.listing(result)])

    def _test_alarm_create_show_query(self, create_params, expected_lines):

        def test(params):
            result = self.aodh('alarm', params=params)
            alarm = self.details_multiple(result)[0]
            for key, value in expected_lines.items():
                self.assertEqual(value, alarm[key])
            return alarm
        alarm = test(create_params)
        params = 'show %s' % alarm['alarm_id']
        test(params)
        self.aodh('alarm', params='delete %s' % alarm['alarm_id'])

    def test_event_alarm_create_show_query(self):
        params = 'create --type event --name alarm-multiple-query --query "traits.project_id=789;traits.resource_id=012"'
        expected_lines = {'query': 'traits.project_id = 789 AND', '': 'traits.resource_id = 012'}
        self._test_alarm_create_show_query(params, expected_lines)
        params = 'create --type event --name alarm-single-query --query "traits.project_id=789"'
        expected_lines = {'query': 'traits.project_id = 789'}
        self._test_alarm_create_show_query(params, expected_lines)
        params = 'create --type event --name alarm-no-query'
        self._test_alarm_create_show_query(params, {'query': ''})

    def test_set_get_alarm_state(self):
        result = self.aodh('alarm', params='create --type event --name alarm_state_test --query "traits.project_id=789;traits.resource_id=012"')
        alarm = self.details_multiple(result)[0]
        alarm_id = alarm['alarm_id']
        result = self.aodh('alarm', params='show %s' % alarm_id)
        alarm_show = self.details_multiple(result)[0]
        self.assertEqual('insufficient data', alarm_show['state'])
        result = self.aodh('alarm', params='state get %s' % alarm_id)
        state_get = self.details_multiple(result)[0]
        self.assertEqual('insufficient data', state_get['state'])
        self.aodh('alarm', params='state set --state ok  %s' % alarm_id)
        result = self.aodh('alarm', params='state get %s' % alarm_id)
        state_get = self.details_multiple(result)[0]
        self.assertEqual('ok', state_get['state'])
        self.aodh('alarm', params='delete %s' % alarm_id)

    def test_update_type_event_composite(self):
        res_id = uuidutils.generate_uuid()
        result = self.aodh(u'alarm', params=u'create --type event --name ev_alarm123')
        alarm = self.details_multiple(result)[0]
        ALARM_ID = alarm['alarm_id']
        self.assertEqual('ev_alarm123', alarm['name'])
        self.assertEqual('*', alarm['event_type'])
        result = self.aodh('alarm', params='update %s --type composite --composite-rule \'{"or":[{"threshold": 0.8, "metric": "cpu_util", "type": "gnocchi_resources_threshold", "resource_type": "generic", "resource_id": "%s", "aggregation_method": "mean"},{"and": [{"threshold": 200, "metric": "disk.iops", "type": "gnocchi_resources_threshold", "resource_type": "generic", "resource_id": "%s", "aggregation_method": "mean"},{"threshold": 1000, "metric": "memory","type": "gnocchi_resources_threshold", "resource_type": "generic", "resource_id": "%s", "aggregation_method": "mean"}]}]}\'' % (ALARM_ID, res_id, res_id, res_id))
        alarm_updated = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
        self.assertEqual('composite', alarm_updated['type'])
        self.assertIn('composite_rule', alarm_updated)
        result = self.aodh('alarm', params='update %s --type event' % ALARM_ID)
        alarm_updated = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
        self.assertEqual('event', alarm_updated['type'])
        self.assertEqual('*', alarm_updated['event_type'])
        result = self.aodh('alarm', params='delete %s' % ALARM_ID)
        self.assertEqual('', result)