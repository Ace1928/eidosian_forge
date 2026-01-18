from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
class ActionExecutionIsolationCLITests(base_v2.MistralClientTestBase):

    def test_action_execution_isolation(self):
        wf = self.workflow_create(self.wf_def)
        wf_exec = self.execution_create(wf[0]['Name'])
        direct_ex_id = self.get_field_value(wf_exec, 'ID')
        self.wait_execution_success(direct_ex_id)
        act_execs = self.mistral_admin('action-execution-list')
        self.assertIn(wf[0]['Name'], [act['Workflow name'] for act in act_execs])
        alt_act_execs = self.mistral_alt_user('action-execution-list')
        self.assertNotIn(wf[0]['Name'], [act['Workflow name'] for act in alt_act_execs])

    def test_get_action_execution_from_another_tenant(self):
        wf = self.workflow_create(self.wf_def)
        ex = self.execution_create(wf[0]['Name'])
        exec_id = self.get_field_value(ex, 'ID')
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'action-execution-get', params=exec_id)