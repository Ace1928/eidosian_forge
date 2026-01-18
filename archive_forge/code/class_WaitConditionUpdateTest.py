import copy
import datetime
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from urllib import parse
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import environment
from heat.engine import node_data
from heat.engine.resources.aws.cfn import wait_condition_handle as aws_wch
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.objects import resource as resource_objects
from heat.tests import common
from heat.tests import utils
class WaitConditionUpdateTest(common.HeatTestCase):

    def create_stack(self, temp=None):
        if temp is None:
            temp = test_template_wc_count
        temp_fmt = template_format.parse(temp)
        template = tmpl.Template(temp_fmt)
        ctx = utils.dummy_context(tenant_id='test_tenant')
        stack = parser.Stack(ctx, 'test_stack', template, disable_rollback=True)
        stack_id = str(uuid.uuid4())
        self.stack_id = stack_id
        with utils.UUIDStub(self.stack_id):
            stack.store()
        with mock.patch.object(aws_wch.WaitConditionHandle, 'get_status') as m_gs:
            m_gs.side_effect = [[], ['SUCCESS'], ['SUCCESS', 'SUCCESS'], ['SUCCESS', 'SUCCESS', 'SUCCESS']]
            stack.create()
            self.assertEqual(4, m_gs.call_count)
        rsrc = stack['WaitForTheHandle']
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        return stack

    def get_stack(self, stack_id):
        ctx = utils.dummy_context(tenant_id='test_tenant')
        stack = parser.Stack.load(ctx, stack_id)
        self.stack_id = stack_id
        return stack

    def test_update(self):
        self.stack = self.create_stack()
        rsrc = self.stack['WaitForTheHandle']
        wait_condition_handle = self.stack['WaitHandle']
        test_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'SUCCESS', 'UniqueId': '1'}
        self._handle_signal(wait_condition_handle, test_metadata, 5)
        uprops = copy.copy(rsrc.properties.data)
        uprops['Count'] = '5'
        update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), uprops)
        updater = scheduler.TaskRunner(rsrc.update, update_snippet)
        updater()
        self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)

    def test_update_restored_from_db(self):
        self.stack = self.create_stack()
        rsrc = self.stack['WaitForTheHandle']
        handle_stack = self.stack
        wait_condition_handle = handle_stack['WaitHandle']
        test_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'SUCCESS', 'UniqueId': '1'}
        self._handle_signal(wait_condition_handle, test_metadata, 2)
        self.stack.store()
        self.stack = self.get_stack(self.stack_id)
        rsrc = self.stack['WaitForTheHandle']
        self._handle_signal(wait_condition_handle, test_metadata, 3)
        uprops = copy.copy(rsrc.properties.data)
        uprops['Count'] = '5'
        update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), uprops)
        stk_defn.update_resource_data(self.stack.defn, 'WaitHandle', self.stack['WaitHandle'].node_data())
        updater = scheduler.TaskRunner(rsrc.update, update_snippet)
        updater()
        self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)

    def _handle_signal(self, rsrc, metadata, times=1):
        for t in range(times):
            metadata['UniqueId'] = metadata['UniqueId'] * 2
            ret = rsrc.handle_signal(metadata)
            self.assertEqual('status:%s reason:%s' % (metadata[rsrc.STATUS], metadata[rsrc.REASON]), ret)

    def test_update_timeout(self):
        self.stack = self.create_stack()
        rsrc = self.stack['WaitForTheHandle']
        now = timeutils.utcnow()
        fake_clock = [now + datetime.timedelta(0, t) for t in (0, 0.001, 0.1, 4.1, 5.1)]
        timeutils.set_time_override(fake_clock)
        self.addCleanup(timeutils.clear_time_override)
        m_gs = self.patchobject(aws_wch.WaitConditionHandle, 'get_status', return_value=[])
        uprops = copy.copy(rsrc.properties.data)
        uprops['Count'] = '5'
        update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), uprops)
        updater = scheduler.TaskRunner(rsrc.update, update_snippet)
        ex = self.assertRaises(exception.ResourceFailure, updater)
        self.assertEqual('WaitConditionTimeout: resources.WaitForTheHandle: 0 of 5 received', str(ex))
        self.assertEqual(5, rsrc.properties['Count'])
        self.assertEqual(2, m_gs.call_count)