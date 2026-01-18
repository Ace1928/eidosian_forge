from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine.resources.openstack.aodh.gnocchi import alarm as gnocchi
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class GnocchiResourcesAlarmTest(common.HeatTestCase):

    def setUp(self):
        super(GnocchiResourcesAlarmTest, self).setUp()
        self.fc = mock.Mock()

    def create_alarm(self):
        self.patchobject(aodh.AodhClientPlugin, '_create').return_value = self.fc
        self.fc.alarm.create.return_value = FakeAodhAlarm
        self.tmpl = template_format.parse(gnocchi_resources_alarm_template)
        self.stack = utils.parse_stack(self.tmpl)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        return gnocchi.AodhGnocchiResourcesAlarm('GnoResAlarm', resource_defns['GnoResAlarm'], self.stack)

    def _check_alarm_create(self):
        expected = {'alarm_actions': [], 'description': u'Do stuff with gnocchi', 'enabled': True, 'insufficient_data_actions': [], 'ok_actions': [], 'name': mock.ANY, 'type': 'gnocchi_resources_threshold', 'repeat_actions': True, 'gnocchi_resources_threshold_rule': {'metric': 'cpu_util', 'aggregation_method': 'mean', 'granularity': 60, 'evaluation_periods': 1, 'threshold': 50, 'resource_type': 'instance', 'resource_id': '5a517ceb-b068-4aca-9eb9-3e4eb9b90d9a', 'comparison_operator': 'gt'}, 'time_constraints': [], 'severity': 'low'}
        self.fc.alarm.create.assert_called_once_with(expected)

    def test_update(self):
        rsrc = self.create_alarm()
        scheduler.TaskRunner(rsrc.create)()
        self._check_alarm_create()
        props = self.tmpl['resources']['GnoResAlarm']['properties']
        props['resource_id'] = 'd3d6c642-921e-4fc2-9c5f-15d9a5afb598'
        update_template = rsrc.t.freeze(properties=props)
        scheduler.TaskRunner(rsrc.update, update_template)()
        self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
        self.fc.alarm.update.assert_called_once_with('foo', {'alarm_actions': [], 'description': u'Do stuff with gnocchi', 'enabled': True, 'insufficient_data_actions': [], 'ok_actions': [], 'repeat_actions': True, 'gnocchi_resources_threshold_rule': {'metric': 'cpu_util', 'aggregation_method': 'mean', 'granularity': 60, 'evaluation_periods': 1, 'threshold': 50, 'resource_type': 'instance', 'resource_id': 'd3d6c642-921e-4fc2-9c5f-15d9a5afb598', 'comparison_operator': 'gt'}, 'time_constraints': [], 'severity': 'low'})

    def _prepare_resource(self, for_check=True):
        snippet = template_format.parse(gnocchi_resources_alarm_template)
        self.stack = utils.parse_stack(snippet)
        res = self.stack['GnoResAlarm']
        if for_check:
            res.state_set(res.CREATE, res.COMPLETE)
        res.client = mock.Mock()
        mock_alarm = mock.Mock(enabled=True, state='ok')
        res.client().alarm.get.return_value = mock_alarm
        return res

    def test_create(self):
        rsrc = self.create_alarm()
        scheduler.TaskRunner(rsrc.create)()
        self._check_alarm_create()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.assertEqual('foo', rsrc.resource_id)

    def test_suspend(self):
        rsrc = self.create_alarm()
        scheduler.TaskRunner(rsrc.create)()
        self._check_alarm_create()
        scheduler.TaskRunner(rsrc.suspend)()
        self.assertEqual((rsrc.SUSPEND, rsrc.COMPLETE), rsrc.state)
        self.fc.alarm.update.assert_called_once_with('foo', {'enabled': False})

    def test_resume(self):
        rsrc = self.create_alarm()
        scheduler.TaskRunner(rsrc.create)()
        self._check_alarm_create()
        rsrc.state_set(rsrc.SUSPEND, rsrc.COMPLETE)
        scheduler.TaskRunner(rsrc.resume)()
        self.assertEqual((rsrc.RESUME, rsrc.COMPLETE), rsrc.state)
        self.fc.alarm.update.assert_called_once_with('foo', {'enabled': True})

    def test_check(self):
        res = self._prepare_resource()
        scheduler.TaskRunner(res.check)()
        self.assertEqual((res.CHECK, res.COMPLETE), res.state)

    def test_check_failure(self):
        res = self._prepare_resource()
        res.client().alarm.get.side_effect = Exception('Boom')
        self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(res.check))
        self.assertEqual((res.CHECK, res.FAILED), res.state)
        self.assertIn('Boom', res.status_reason)

    def test_show_resource(self):
        res = self._prepare_resource(for_check=False)
        res.client().alarm.create.return_value = FakeAodhAlarm
        res.client().alarm.get.return_value = FakeAodhAlarm
        scheduler.TaskRunner(res.create)()
        self.assertEqual(FakeAodhAlarm, res.FnGetAtt('show'))

    def test_gnocchi_alarm_live_state(self):
        snippet = template_format.parse(gnocchi_resources_alarm_template)
        self.stack = utils.parse_stack(snippet)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        self.rsrc_defn = resource_defns['GnoResAlarm']
        self.client = mock.Mock()
        self.patchobject(gnocchi.AodhGnocchiResourcesAlarm, 'client', return_value=self.client)
        alarm_res = gnocchi.AodhGnocchiResourcesAlarm('alarm', self.rsrc_defn, self.stack)
        alarm_res.create()
        value = {'description': 'Do stuff with gnocchi', 'alarm_actions': [], 'time_constraints': [], 'gnocchi_resources_threshold_rule': {'resource_id': '5a517ceb-b068-4aca-9eb9-3e4eb9b90d9a', 'metric': 'cpu_util', 'evaluation_periods': 1, 'aggregation_method': 'mean', 'granularity': 60, 'threshold': 50, 'comparison_operator': 'gt', 'resource_type': 'instance'}}
        self.client.alarm.get.return_value = value
        expected_data = {'description': 'Do stuff with gnocchi', 'alarm_actions': [], 'resource_id': '5a517ceb-b068-4aca-9eb9-3e4eb9b90d9a', 'metric': 'cpu_util', 'evaluation_periods': 1, 'aggregation_method': 'mean', 'granularity': 60, 'threshold': 50, 'comparison_operator': 'gt', 'resource_type': 'instance', 'insufficient_data_actions': None, 'enabled': None, 'ok_actions': None, 'repeat_actions': None, 'severity': None}
        reality = alarm_res.get_live_state(alarm_res.properties)
        self.assertEqual(expected_data, reality)