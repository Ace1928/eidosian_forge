import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
class AodhClientGnocchiRulesTest(base.ClientTestBase):

    def test_gnocchi_resources_threshold_scenario(self):
        PROJECT_ID = uuidutils.generate_uuid()
        RESOURCE_ID = uuidutils.generate_uuid()
        req = requests.post(os.environ.get('GNOCCHI_ENDPOINT') + '/v1/resource/generic', headers={'X-Auth-Token': self.get_token()}, json={'id': RESOURCE_ID})
        self.assertEqual(201, req.status_code)
        result = self.aodh(u'alarm', params=u'create --type gnocchi_resources_threshold --name alarm_gn1 --metric cpu_util --threshold 80 --resource-id %s --resource-type generic --aggregation-method last --project-id %s' % (RESOURCE_ID, PROJECT_ID))
        alarm = self.details_multiple(result)[0]
        ALARM_ID = alarm['alarm_id']
        self.assertEqual('alarm_gn1', alarm['name'])
        self.assertEqual('cpu_util', alarm['metric'])
        self.assertEqual('80.0', alarm['threshold'])
        self.assertEqual('last', alarm['aggregation_method'])
        self.assertEqual(RESOURCE_ID, alarm['resource_id'])
        self.assertEqual('generic', alarm['resource_type'])
        result = self.aodh(u'alarm', params=u"create --type gnocchi_resources_threshold --name alarm_tc --metric cpu_util --threshold 80 --resource-id %s --resource-type generic --aggregation-method last --project-id %s --time-constraint name=cons1;start='0 11 * * *';duration=300 --time-constraint name=cons2;start='0 23 * * *';duration=600 " % (RESOURCE_ID, PROJECT_ID))
        alarm = self.details_multiple(result)[0]
        self.assertEqual('alarm_tc', alarm['name'])
        self.assertEqual('80.0', alarm['threshold'])
        self.assertIsNotNone(alarm['time_constraints'])
        self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'create --type gnocchi_resources_threshold --name alarm1 --metric cpu_util --resource-id %s --resource-type generic --aggregation-method last --project-id %s' % (RESOURCE_ID, PROJECT_ID))
        result = self.aodh('alarm', params='update %s --severity critical --threshold 90' % ALARM_ID)
        alarm_updated = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
        self.assertEqual('critical', alarm_updated['severity'])
        self.assertEqual('90.0', alarm_updated['threshold'])
        result = self.aodh('alarm', params='show %s' % ALARM_ID)
        alarm_show = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
        self.assertEqual(PROJECT_ID, alarm_show['project_id'])
        self.assertEqual('alarm_gn1', alarm_show['name'])
        self.assertEqual('cpu_util', alarm_show['metric'])
        self.assertEqual('90.0', alarm_show['threshold'])
        self.assertEqual('critical', alarm_show['severity'])
        self.assertEqual('last', alarm_show['aggregation_method'])
        self.assertEqual('generic', alarm_show['resource_type'])
        result = self.aodh('alarm', params='show --name alarm_gn1')
        alarm_show = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
        self.assertEqual(PROJECT_ID, alarm_show['project_id'])
        self.assertEqual('alarm_gn1', alarm_show['name'])
        self.assertEqual('cpu_util', alarm_show['metric'])
        self.assertEqual('90.0', alarm_show['threshold'])
        self.assertEqual('critical', alarm_show['severity'])
        self.assertEqual('last', alarm_show['aggregation_method'])
        self.assertEqual('generic', alarm_show['resource_type'])
        self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'show %s --name alarm_gn1' % ALARM_ID)
        result = self.aodh('alarm', params='list --filter all_projects=true')
        self.assertIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])
        output_colums = ['alarm_id', 'type', 'name', 'state', 'severity', 'enabled']
        for alarm_list in self.parser.listing(result):
            self.assertEqual(sorted(output_colums), sorted(alarm_list.keys()))
            if alarm_list['alarm_id'] == ALARM_ID:
                self.assertEqual('alarm_gn1', alarm_list['name'])
        result = self.aodh('alarm', params='list --filter all_projects=true --limit 1')
        alarm_list = self.parser.listing(result)
        self.assertEqual(1, len(alarm_list))
        result = self.aodh('alarm', params='list --filter all_projects=true --sort name:asc')
        names = [r['name'] for r in self.parser.listing(result)]
        sorted_name = sorted(names)
        self.assertEqual(sorted_name, names)
        result = self.aodh(u'alarm', params=u'create --type gnocchi_resources_threshold --name alarm_th --metric cpu_util --threshold 80 --resource-id %s --resource-type generic --aggregation-method last --project-id %s ' % (RESOURCE_ID, PROJECT_ID))
        created_alarm_id = self.details_multiple(result)[0]['alarm_id']
        result = self.aodh('alarm', params='list --filter all_projects=true --sort name:asc --sort alarm_id:asc')
        alarm_list = self.parser.listing(result)
        ids_with_same_name = []
        names = []
        for alarm in alarm_list:
            names.append(['alarm_name'])
            if alarm['name'] == 'alarm_th':
                ids_with_same_name.append(alarm['alarm_id'])
        sorted_ids = sorted(ids_with_same_name)
        sorted_names = sorted(names)
        self.assertEqual(sorted_names, names)
        self.assertEqual(sorted_ids, ids_with_same_name)
        result = self.aodh('alarm', params='list --filter all_projects=true --sort name:desc --marker %s' % created_alarm_id)
        self.assertIn('alarm_tc', [r['name'] for r in self.parser.listing(result)])
        self.aodh('alarm', params='delete %s' % created_alarm_id)
        result = self.aodh('alarm', params='list --query project_id=%s' % PROJECT_ID)
        alarm_list = self.parser.listing(result)[0]
        self.assertEqual(ALARM_ID, alarm_list['alarm_id'])
        self.assertEqual('alarm_gn1', alarm_list['name'])
        result = self.aodh('alarm', params='delete %s' % ALARM_ID)
        self.assertEqual('', result)
        result = self.aodh('alarm', params='show %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
        expected = 'Alarm %s not found (HTTP 404)' % ALARM_ID
        self.assertFirstLineStartsWith(result.splitlines(), expected)
        result = self.aodh('alarm', params='delete %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
        self.assertFirstLineStartsWith(result.splitlines(), expected)
        result = self.aodh('alarm', params='list')
        self.assertNotIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])

    def test_gnocchi_aggr_by_resources_scenario(self):
        result = self.aodh(u'alarm', params=u'create --type gnocchi_aggregation_by_resources_threshold --name alarm1 --metric cpu --threshold 80 --query \'{"=": {"creator": "cr3at0r"}}\' --resource-type generic --aggregation-method mean ')
        alarm = self.details_multiple(result)[0]
        ALARM_ID = alarm['alarm_id']
        self.assertEqual('alarm1', alarm['name'])
        self.assertEqual('cpu', alarm['metric'])
        self.assertEqual('80.0', alarm['threshold'])
        self.assertEqual('mean', alarm['aggregation_method'])
        self.assertEqual('generic', alarm['resource_type'])
        self.assertEqual('{"=": {"creator": "cr3at0r"}}', alarm['query'])
        self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'create --type gnocchi_aggregation_by_resources_threshold --name alarm1 --metric cpu --query \'{"=": {"creator": "cr3at0r"}}\' --resource-type generic --aggregation-method mean ')
        result = self.aodh('alarm', params='update %s --severity critical --threshold 90' % ALARM_ID)
        alarm_updated = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
        self.assertEqual('critical', alarm_updated['severity'])
        self.assertEqual('90.0', alarm_updated['threshold'])
        result = self.aodh('alarm', params='show %s' % ALARM_ID)
        alarm_show = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
        self.assertEqual('alarm1', alarm_show['name'])
        self.assertEqual('cpu', alarm_show['metric'])
        self.assertEqual('90.0', alarm_show['threshold'])
        self.assertEqual('critical', alarm_show['severity'])
        self.assertEqual('mean', alarm_show['aggregation_method'])
        self.assertEqual('generic', alarm_show['resource_type'])
        result = self.aodh('alarm', params='list --filter all_projects=true')
        self.assertIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])
        output_colums = ['alarm_id', 'type', 'name', 'state', 'severity', 'enabled']
        for alarm_list in self.parser.listing(result):
            self.assertEqual(sorted(output_colums), sorted(alarm_list.keys()))
            if alarm_list['alarm_id'] == ALARM_ID:
                self.assertEqual('alarm1', alarm_list['name'])
        result = self.aodh('alarm', params='delete %s' % ALARM_ID)
        self.assertEqual('', result)
        result = self.aodh('alarm', params='show %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
        expected = 'Alarm %s not found (HTTP 404)' % ALARM_ID
        self.assertFirstLineStartsWith(result.splitlines(), expected)
        result = self.aodh('alarm', params='delete %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
        self.assertFirstLineStartsWith(result.splitlines(), expected)
        result = self.aodh('alarm', params='list')
        self.assertNotIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])

    def test_gnocchi_aggr_by_metrics_scenario(self):
        PROJECT_ID = uuidutils.generate_uuid()
        METRIC1 = 'cpu'
        METRIC2 = 'cpu_util'
        result = self.aodh(u'alarm', params=u'create --type gnocchi_aggregation_by_metrics_threshold --name alarm1 --metrics %s --metric %s --threshold 80 --aggregation-method last --project-id %s' % (METRIC1, METRIC2, PROJECT_ID))
        alarm = self.details_multiple(result)[0]
        ALARM_ID = alarm['alarm_id']
        self.assertEqual('alarm1', alarm['name'])
        metrics = "['cpu', 'cpu_util']"
        self.assertEqual(metrics, alarm['metrics'])
        self.assertEqual('80.0', alarm['threshold'])
        self.assertEqual('last', alarm['aggregation_method'])
        self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'create --type gnocchi_aggregation_by_metrics_threshold --name alarm1 --metrics %s --metrics %s --aggregation-method last --project-id %s' % (METRIC1, METRIC2, PROJECT_ID))
        result = self.aodh('alarm', params='update %s --severity critical --threshold 90' % ALARM_ID)
        alarm_updated = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
        self.assertEqual('critical', alarm_updated['severity'])
        self.assertEqual('90.0', alarm_updated['threshold'])
        result = self.aodh('alarm', params='show %s' % ALARM_ID)
        alarm_show = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
        self.assertEqual(PROJECT_ID, alarm_show['project_id'])
        self.assertEqual('alarm1', alarm_show['name'])
        self.assertEqual(metrics, alarm_show['metrics'])
        self.assertEqual('90.0', alarm_show['threshold'])
        self.assertEqual('critical', alarm_show['severity'])
        self.assertEqual('last', alarm_show['aggregation_method'])
        result = self.aodh('alarm', params='list --filter all_projects=true')
        self.assertIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])
        for alarm_list in self.parser.listing(result):
            if alarm_list['alarm_id'] == ALARM_ID:
                self.assertEqual('alarm1', alarm_list['name'])
        result = self.aodh('alarm', params='list --query project_id=%s' % PROJECT_ID)
        alarm_list = self.parser.listing(result)[0]
        self.assertEqual(ALARM_ID, alarm_list['alarm_id'])
        self.assertEqual('alarm1', alarm_list['name'])
        result = self.aodh('alarm', params='delete %s' % ALARM_ID)
        self.assertEqual('', result)
        result = self.aodh('alarm', params='show %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
        expected = 'Alarm %s not found (HTTP 404)' % ALARM_ID
        self.assertFirstLineStartsWith(result.splitlines(), expected)
        result = self.aodh('alarm', params='delete %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
        self.assertFirstLineStartsWith(result.splitlines(), expected)
        result = self.aodh('alarm', params='list')
        output_colums = ['alarm_id', 'type', 'name', 'state', 'severity', 'enabled']
        for alarm_list in self.parser.listing(result):
            self.assertEqual(sorted(output_colums), sorted(alarm_list.keys()))
        self.assertNotIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])

    def test_update_gnresthr_gnaggrresthr(self):
        RESOURCE_ID = uuidutils.generate_uuid()
        result = self.aodh(u'alarm', params=u'create --type gnocchi_resources_threshold --name alarm_gn123 --metric cpu_util --resource-id %s --threshold 80 --resource-type generic --aggregation-method last ' % RESOURCE_ID)
        alarm = self.details_multiple(result)[0]
        ALARM_ID = alarm['alarm_id']
        self.assertEqual('alarm_gn123', alarm['name'])
        self.assertEqual('cpu_util', alarm['metric'])
        self.assertEqual('80.0', alarm['threshold'])
        self.assertEqual('last', alarm['aggregation_method'])
        self.assertEqual('generic', alarm['resource_type'])
        result = self.aodh('alarm', params='update %s --type gnocchi_aggregation_by_resources_threshold --metric cpu --threshold 90 --query \'{"=": {"creator": "cr3at0r"}}\' --resource-type generic --aggregation-method mean ' % ALARM_ID)
        alarm_updated = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
        self.assertEqual('cpu', alarm_updated['metric'])
        self.assertEqual('90.0', alarm_updated['threshold'])
        self.assertEqual('mean', alarm_updated['aggregation_method'])
        self.assertEqual('generic', alarm_updated['resource_type'])
        self.assertEqual('{"=": {"creator": "cr3at0r"}}', alarm_updated['query'])
        self.assertEqual('gnocchi_aggregation_by_resources_threshold', alarm_updated['type'])
        result = self.aodh('alarm', params='update %s --type gnocchi_resources_threshold --metric cpu_util --resource-id %s --threshold 80 --resource-type generic --aggregation-method last ' % (ALARM_ID, RESOURCE_ID))
        alarm_updated = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
        self.assertEqual('cpu_util', alarm_updated['metric'])
        self.assertEqual('80.0', alarm_updated['threshold'])
        self.assertEqual('last', alarm_updated['aggregation_method'])
        self.assertEqual('generic', alarm_updated['resource_type'])
        self.assertEqual('gnocchi_resources_threshold', alarm_updated['type'])
        result = self.aodh('alarm', params='delete %s' % ALARM_ID)
        self.assertEqual('', result)

    def test_update_gnaggrresthr_gnaggrmetricthr(self):
        METRIC1 = 'cpu'
        METRIC2 = 'cpu_util'
        result = self.aodh(u'alarm', params=u'create --type gnocchi_aggregation_by_resources_threshold --name alarm123 --metric cpu --threshold 80 --query \'{"=": {"creator": "cr3at0r"}}\' --resource-type generic --aggregation-method mean ')
        alarm = self.details_multiple(result)[0]
        ALARM_ID = alarm['alarm_id']
        self.assertEqual('alarm123', alarm['name'])
        self.assertEqual('cpu', alarm['metric'])
        self.assertEqual('80.0', alarm['threshold'])
        self.assertEqual('mean', alarm['aggregation_method'])
        self.assertEqual('generic', alarm['resource_type'])
        self.assertEqual('{"=": {"creator": "cr3at0r"}}', alarm['query'])
        result = self.aodh('alarm', params='update %s --type gnocchi_aggregation_by_metrics_threshold --metrics %s --metrics %s --threshold 80 --aggregation-method last' % (ALARM_ID, METRIC1, METRIC2))
        alarm_updated = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
        metrics = "['cpu', 'cpu_util']"
        self.assertEqual(metrics, alarm_updated['metrics'])
        self.assertEqual('80.0', alarm_updated['threshold'])
        self.assertEqual('last', alarm_updated['aggregation_method'])
        self.assertEqual('gnocchi_aggregation_by_metrics_threshold', alarm_updated['type'])
        result = self.aodh('alarm', params='update %s --type gnocchi_aggregation_by_resources_threshold --metric cpu --threshold 80 --query \'{"=": {"creator": "cr3at0r"}}\' --resource-type generic --aggregation-method mean ' % ALARM_ID)
        alarm_updated = self.details_multiple(result)[0]
        self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
        self.assertEqual('cpu', alarm_updated['metric'])
        self.assertEqual('80.0', alarm_updated['threshold'])
        self.assertEqual('mean', alarm_updated['aggregation_method'])
        self.assertEqual('generic', alarm_updated['resource_type'])
        self.assertEqual('{"=": {"creator": "cr3at0r"}}', alarm_updated['query'])
        self.assertEqual('gnocchi_aggregation_by_resources_threshold', alarm_updated['type'])
        result = self.aodh('alarm', params='delete %s' % ALARM_ID)
        self.assertEqual('', result)