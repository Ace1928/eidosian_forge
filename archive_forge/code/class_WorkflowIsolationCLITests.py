from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
class WorkflowIsolationCLITests(base_v2.MistralClientTestBase):

    def test_workflow_name_uniqueness(self):
        self.workflow_create(self.wf_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-create', params='{0}'.format(self.wf_def))
        self.workflow_create(self.wf_def, admin=False)
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'workflow-create', params='{0}'.format(self.wf_def))

    def test_wf_isolation(self):
        wf = self.workflow_create(self.wf_def)
        wfs = self.mistral_admin('workflow-list')
        self.assertIn(wf[0]['Name'], [w['Name'] for w in wfs])
        alt_wfs = self.mistral_alt_user('workflow-list')
        self.assertNotIn(wf[0]['Name'], [w['Name'] for w in alt_wfs])

    def test_get_wf_from_another_tenant(self):
        wf = self.workflow_create(self.wf_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'workflow-get', params=wf[0]['ID'])

    def test_create_public_workflow(self):
        wf = self.workflow_create(self.wf_def, scope='public')
        same_wf = self.mistral_alt_user('workflow-get', params=wf[0]['Name'])
        self.assertEqual(wf[0]['Name'], self.get_field_value(same_wf, 'Name'))

    def test_delete_wf_from_another_tenant(self):
        wf = self.workflow_create(self.wf_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'workflow-delete', params=wf[0]['ID'])