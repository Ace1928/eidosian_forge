import time
from oslo_concurrency import processutils
from heat_integrationtests.functional import functional_base
class PurgeTest(functional_base.FunctionalTestsBase):
    template = '\nheat_template_version: 2014-10-16\nparameters:\nresources:\n  test_resource:\n    type: OS::Heat::TestResource\n'

    def test_purge(self):
        stack_identifier = self.stack_create(template=self.template)
        self._stack_delete(stack_identifier)
        stacks = dict(((stack.id, stack) for stack in self.client.stacks.list(show_deleted=True)))
        self.assertIn(stack_identifier.split('/')[1], stacks)
        time.sleep(1)
        cmd = 'heat-manage purge_deleted 0'
        processutils.execute(cmd, shell=True)
        stacks = dict(((stack.id, stack) for stack in self.client.stacks.list(show_deleted=True)))
        self.assertNotIn(stack_identifier.split('/')[1], stacks)
        stack_identifier = self.stack_create(template=self.template, tags='foo,bar')
        self._stack_delete(stack_identifier)
        time.sleep(1)
        cmd = 'heat-manage purge_deleted 0'
        processutils.execute(cmd, shell=True)
        stacks = dict(((stack.id, stack) for stack in self.client.stacks.list(show_deleted=True)))
        self.assertNotIn(stack_identifier.split('/')[1], stacks)