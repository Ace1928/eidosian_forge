from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def _update_shared_workflow(self, new_status='accepted'):
    member = self.workflow_member_create(self.wf[0]['ID'])
    status = self.get_field_value(member, 'Status')
    self.assertEqual('pending', status)
    cmd_param = '%s workflow --status %s --member-id %s' % (self.wf[0]['ID'], new_status, self.get_project_id('alt_demo'))
    member = self.mistral_alt_user('member-update', params=cmd_param)
    status = self.get_field_value(member, 'Status')
    self.assertEqual(new_status, status)