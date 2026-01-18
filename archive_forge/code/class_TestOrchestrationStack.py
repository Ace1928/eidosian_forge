from unittest import mock
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.orchestration.v1 import _proxy
from openstack.orchestration.v1 import resource
from openstack.orchestration.v1 import software_config as sc
from openstack.orchestration.v1 import software_deployment as sd
from openstack.orchestration.v1 import stack
from openstack.orchestration.v1 import stack_environment
from openstack.orchestration.v1 import stack_event
from openstack.orchestration.v1 import stack_files
from openstack.orchestration.v1 import stack_template
from openstack.orchestration.v1 import template
from openstack import proxy
from openstack.tests.unit import test_proxy_base
class TestOrchestrationStack(TestOrchestrationProxy):

    def test_create_stack(self):
        self.verify_create(self.proxy.create_stack, stack.Stack)

    def test_create_stack_preview(self):
        self.verify_create(self.proxy.create_stack, stack.Stack, method_kwargs={'preview': True, 'x': 1, 'y': 2, 'z': 3}, expected_kwargs={'x': 1, 'y': 2, 'z': 3})

    def test_find_stack(self):
        self.verify_find(self.proxy.find_stack, stack.Stack, expected_kwargs={'resolve_outputs': True})

    def test_stacks(self):
        self.verify_list(self.proxy.stacks, stack.Stack)

    def test_get_stack(self):
        self.verify_get(self.proxy.get_stack, stack.Stack, method_kwargs={'resolve_outputs': False}, expected_kwargs={'resolve_outputs': False})
        self.verify_get_overrided(self.proxy, stack.Stack, 'openstack.orchestration.v1.stack.Stack')

    def test_update_stack(self):
        self._verify('openstack.orchestration.v1.stack.Stack.update', self.proxy.update_stack, expected_result='result', method_args=['stack'], method_kwargs={'preview': False}, expected_args=[self.proxy, False])

    def test_update_stack_preview(self):
        self._verify('openstack.orchestration.v1.stack.Stack.update', self.proxy.update_stack, expected_result='result', method_args=['stack'], method_kwargs={'preview': True}, expected_args=[self.proxy, True])

    def test_abandon_stack(self):
        self._verify('openstack.orchestration.v1.stack.Stack.abandon', self.proxy.abandon_stack, expected_result='result', method_args=['stack'], expected_args=[self.proxy])

    @mock.patch.object(stack.Stack, 'find')
    def test_export_stack_with_identity(self, mock_find):
        stack_id = '1234'
        stack_name = 'test_stack'
        stk = stack.Stack(id=stack_id, name=stack_name)
        mock_find.return_value = stk
        self._verify('openstack.orchestration.v1.stack.Stack.export', self.proxy.export_stack, method_args=['IDENTITY'], expected_args=[self.proxy])
        mock_find.assert_called_once_with(mock.ANY, 'IDENTITY', ignore_missing=False)

    def test_export_stack_with_object(self):
        stack_id = '1234'
        stack_name = 'test_stack'
        stk = stack.Stack(id=stack_id, name=stack_name)
        self._verify('openstack.orchestration.v1.stack.Stack.export', self.proxy.export_stack, method_args=[stk], expected_args=[self.proxy])

    def test_suspend_stack(self):
        self._verify('openstack.orchestration.v1.stack.Stack.suspend', self.proxy.suspend_stack, method_args=['stack'], expected_args=[self.proxy])

    def test_resume_stack(self):
        self._verify('openstack.orchestration.v1.stack.Stack.resume', self.proxy.resume_stack, method_args=['stack'], expected_args=[self.proxy])

    def test_delete_stack(self):
        self.verify_delete(self.proxy.delete_stack, stack.Stack, False)

    def test_delete_stack_ignore(self):
        self.verify_delete(self.proxy.delete_stack, stack.Stack, True)

    @mock.patch.object(stack.Stack, 'check')
    def test_check_stack_with_stack_object(self, mock_check):
        stk = stack.Stack(id='FAKE_ID')
        res = self.proxy.check_stack(stk)
        self.assertIsNone(res)
        mock_check.assert_called_once_with(self.proxy)

    @mock.patch.object(stack.Stack, 'existing')
    def test_check_stack_with_stack_ID(self, mock_stack):
        stk = mock.Mock()
        mock_stack.return_value = stk
        res = self.proxy.check_stack('FAKE_ID')
        self.assertIsNone(res)
        mock_stack.assert_called_once_with(id='FAKE_ID')
        stk.check.assert_called_once_with(self.proxy)