from tempest.lib.common.utils import data_utils as utils
from heatclient.tests.functional import config
from heatclient.tests.functional.osc.v1 import base
class OpenStackClientStackTest(base.OpenStackClientTestBase):
    """Basic stack operation tests for Openstack CLI client heat plugin.

    Basic smoke test for the openstack CLI stack commands.
    """

    def setUp(self):
        super(OpenStackClientStackTest, self).setUp()
        self.stack_name = utils.rand_name(name='test-stack')

    def _stack_create_minimal(self, from_url=False):
        if from_url:
            template = config.HEAT_MINIMAL_HOT_TEMPLATE_URL
        else:
            template = self.get_template_path('heat_minimal_hot.yaml')
        parameters = ['test_client_name=test_client_name']
        return self._stack_create(self.stack_name, template=template, parameters=parameters)

    def test_stack_create_minimal_from_file(self):
        stack = self._stack_create_minimal()
        self.assertEqual(self.stack_name, stack['stack_name'])
        self.assertEqual('CREATE_COMPLETE', stack['stack_status'])

    def test_stack_create_minimal_from_url(self):
        stack = self._stack_create_minimal(from_url=True)
        self.assertEqual(self.stack_name, stack['stack_name'])
        self.assertEqual('CREATE_COMPLETE', stack['stack_status'])

    def test_stack_suspend_resume(self):
        stack = self._stack_create_minimal()
        stack = self._stack_suspend(stack['id'])
        self.assertEqual(self.stack_name, stack['stack_name'])
        self.assertEqual('SUSPEND_COMPLETE', stack['stack_status'])
        stack = self._stack_resume(stack['id'])
        self.assertEqual(self.stack_name, stack['stack_name'])
        self.assertEqual('RESUME_COMPLETE', stack['stack_status'])

    def test_stack_snapshot_create_restore(self):
        snapshot_name = utils.rand_name(name='test-stack-snapshot')
        stack = self._stack_create_minimal()
        snapshot = self._stack_snapshot_create(stack['id'], snapshot_name)
        self.assertEqual(snapshot_name, snapshot['name'])
        self._stack_snapshot_restore(stack['id'], snapshot['id'])

    def test_stack_delete(self):
        stack = self._stack_create_minimal()
        self._stack_delete(stack['id'])
        stacks_raw = self.openstack('stack list')
        self.assertNotIn(stack['id'], stacks_raw)

    def test_stack_snapshot_delete(self):
        snapshot_name = utils.rand_name(name='test-stack-snapshot')
        stack = self._stack_create_minimal()
        snapshot = self._stack_snapshot_create(stack['id'], snapshot_name)
        self._stack_snapshot_delete(stack['id'], snapshot['id'])
        stacks_raw = self.openstack('stack snapshot list' + ' ' + self.stack_name)
        self.assertNotIn(snapshot['id'], stacks_raw)

    def test_stack_show(self):
        stack = self._stack_create_minimal()
        stack_info = self._stack_show(stack['id'])
        stack_props = {k: v for k, v in stack_info.items() if k in stack.keys()}
        self.assertEqual(stack, stack_props)