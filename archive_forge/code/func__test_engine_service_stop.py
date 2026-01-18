import datetime
from unittest import mock
from oslo_config import cfg
from oslo_utils import timeutils
from heat.common import context
from heat.common import service_utils
from heat.engine import service
from heat.engine import worker
from heat.objects import service as service_objects
from heat.rpc import worker_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def _test_engine_service_stop(self, service_delete_method, admin_context_method):
    cfg.CONF.set_default('periodic_interval', 60)
    self.patchobject(self.eng, 'service_manage_cleanup')
    self.patchobject(self.eng, 'reset_stack_status')
    self.patchobject(self.eng, 'service_manage_report')
    self.eng.start()
    dtg1 = tools.DummyThreadGroup()
    dtg2 = tools.DummyThreadGroup()
    self.eng.thread_group_mgr.groups['sample-uuid1'] = dtg1
    self.eng.thread_group_mgr.groups['sample-uuid2'] = dtg2
    self.eng.service_id = 'sample-service-uuid'
    self.patchobject(self.eng.manage_thread_grp, 'stop', new=mock.Mock(wraps=self.eng.manage_thread_grp.stop))
    self.patchobject(self.eng, '_stop_rpc_server', new=mock.Mock(wraps=self.eng._stop_rpc_server))
    orig_stop = self.eng.thread_group_mgr.stop
    with mock.patch.object(self.eng.thread_group_mgr, 'stop') as stop:
        stop.side_effect = orig_stop
        self.eng.stop()
        self.eng._stop_rpc_server.assert_called_once_with()
        if cfg.CONF.convergence_engine:
            self.eng.worker_service.stop.assert_called_once_with()
        calls = [mock.call('sample-uuid1', True), mock.call('sample-uuid2', True)]
        self.eng.thread_group_mgr.stop.assert_has_calls(calls, True)
        self.eng.manage_thread_grp.stop.assert_called_with()
        admin_context_method.assert_called_once_with()
        ctxt = admin_context_method.return_value
        service_delete_method.assert_called_once_with(ctxt, self.eng.service_id)