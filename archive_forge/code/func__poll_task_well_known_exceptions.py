from datetime import datetime
from unittest import mock
from eventlet import greenthread
from oslo_context import context
import suds
from oslo_vmware import api
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def _poll_task_well_known_exceptions(self, fault, expected_exception):
    api_session = self._create_api_session(False)

    def fake_invoke_api(self, module, method, *args, **kwargs):
        task_info = mock.Mock()
        task_info.progress = -1
        task_info.state = 'error'
        error = mock.Mock()
        error.localizedMessage = 'Error message'
        error_fault = mock.Mock()
        error_fault.__class__.__name__ = fault
        error.fault = error_fault
        task_info.error = error
        return task_info
    with mock.patch.object(api_session, 'invoke_api', fake_invoke_api):
        fake_task = vim_util.get_moref('Task', 'task-1')
        ctx = mock.Mock()
        self.assertRaises(expected_exception, api_session._poll_task, fake_task, ctx)