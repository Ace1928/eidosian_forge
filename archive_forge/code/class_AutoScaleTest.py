from unittest import mock
from oslo_utils import timeutils
from heat.common import timeutils as heat_timeutils
from heat.engine import notification
from heat.tests import common
from heat.tests import utils
class AutoScaleTest(common.HeatTestCase):

    def setUp(self):
        super(AutoScaleTest, self).setUp()
        self.ctx = utils.dummy_context(user_id='test_user_id')

    def _mock_stack(self):
        created_time = timeutils.utcnow()
        st = mock.Mock()
        st.state = ('x', 'f')
        st.status = st.state[0]
        st.action = st.state[1]
        st.name = 'fred'
        st.status_reason = 'this is why'
        st.created_time = created_time
        st.context = self.ctx
        st.id = 'hay-are-en'
        updated_time = timeutils.utcnow()
        st.updated_time = updated_time
        st.tags = ['tag1', 'tag2']
        st.t = mock.MagicMock()
        st.t.__getitem__.return_value = 'for test'
        st.t.DESCRIPTION = 'description'
        return st

    def test_send(self):
        stack = self._mock_stack()
        notify = self.patchobject(notification, 'notify')
        notification.autoscaling.send(stack, adjustment='x', adjustment_type='y', capacity='5', groupname='c', message='fred', suffix='the-end')
        notify.assert_called_once_with(self.ctx, 'autoscaling.the-end', 'INFO', {'state_reason': 'this is why', 'user_id': 'test_username', 'username': 'test_username', 'user_identity': 'test_user_id', 'stack_identity': 'hay-are-en', 'stack_name': 'fred', 'tenant_id': 'test_tenant_id', 'create_at': heat_timeutils.isotime(stack.created_time), 'description': 'for test', 'tags': ['tag1', 'tag2'], 'updated_at': heat_timeutils.isotime(stack.updated_time), 'state': 'x_f', 'adjustment_type': 'y', 'groupname': 'c', 'capacity': '5', 'message': 'fred', 'adjustment': 'x'})

    def test_send_error(self):
        stack = self._mock_stack()
        notify = self.patchobject(notification, 'notify')
        notification.autoscaling.send(stack, adjustment='x', adjustment_type='y', capacity='5', groupname='c', suffix='error')
        notify.assert_called_once_with(self.ctx, 'autoscaling.error', 'ERROR', {'state_reason': 'this is why', 'user_id': 'test_username', 'username': 'test_username', 'user_identity': 'test_user_id', 'stack_identity': 'hay-are-en', 'stack_name': 'fred', 'tenant_id': 'test_tenant_id', 'create_at': heat_timeutils.isotime(stack.created_time), 'description': 'for test', 'tags': ['tag1', 'tag2'], 'updated_at': heat_timeutils.isotime(stack.updated_time), 'state': 'x_f', 'adjustment_type': 'y', 'groupname': 'c', 'capacity': '5', 'message': 'error', 'adjustment': 'x'})