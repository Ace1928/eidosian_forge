import copy
import json
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine.clients.os import octavia
from heat.engine import resource
from heat.engine.resources.openstack.aodh import alarm
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
class LBMemberHealthAlarmTest(common.HeatTestCase):

    def setUp(self):
        super(LBMemberHealthAlarmTest, self).setUp()
        self.fa = mock.Mock()
        self.patchobject(octavia.OctaviaClientPlugin, 'get_pool').return_value = '9999'

    def create_stack(self, template=None):
        if template is None:
            template = lbmemberhealth_alarm_template
        temp = template_format.parse(template)
        template = tmpl.Template(temp)
        ctx = utils.dummy_context()
        ctx.tenant = 'test_tenant'
        stack = parser.Stack(ctx, utils.random_name(), template, disable_rollback=True)
        stack.store()
        self.patchobject(aodh.AodhClientPlugin, '_create').return_value = self.fa
        self.patchobject(self.fa.alarm, 'create').return_value = FakeAodhAlarm
        return stack

    def _prepare_resource(self, for_check=True):
        snippet = template_format.parse(lbmemberhealth_alarm_template)
        self.stack = utils.parse_stack(snippet)
        res = self.stack['test_loadbalancer_member_health_alarm']
        if for_check:
            res.state_set(res.CREATE, res.COMPLETE)
        res.client = mock.Mock()
        mock_alarm = mock.Mock(enabled=True, state='ok')
        res.client().alarm.get.return_value = mock_alarm
        return res

    def test_delete(self):
        test_stack = self.create_stack()
        rsrc = test_stack['test_loadbalancer_member_health_alarm']
        self.patchobject(aodh.AodhClientPlugin, 'client', return_value=self.fa)
        self.patchobject(self.fa.alarm, 'delete')
        rsrc.resource_id = '12345'
        self.assertEqual('12345', rsrc.handle_delete())
        self.assertEqual(1, self.fa.alarm.delete.call_count)

    def test_check(self):
        res = self._prepare_resource()
        scheduler.TaskRunner(res.check)()
        self.assertEqual((res.CHECK, res.COMPLETE), res.state)

    def test_check_alarm_failure(self):
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

    def test_update(self):
        test_stack = self.create_stack()
        update_mock = self.patchobject(self.fa.alarm, 'update')
        test_stack.create()
        rsrc = test_stack['test_loadbalancer_member_health_alarm']
        update_props = copy.deepcopy(rsrc.properties.data)
        update_props.update({'enabled': True, 'description': '', 'insufficient_data_actions': [], 'alarm_actions': [], 'ok_actions': ['signal_handler'], 'pool': '0000', 'autoscaling_group_id': '2222'})
        snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), update_props)
        scheduler.TaskRunner(rsrc.update, snippet)()
        self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
        self.assertEqual(1, update_mock.call_count)