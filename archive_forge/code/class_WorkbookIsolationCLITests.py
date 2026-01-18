from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
class WorkbookIsolationCLITests(base_v2.MistralClientTestBase):

    def test_workbook_name_uniqueness(self):
        self.workbook_create(self.wb_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-create', params='{0}'.format(self.wb_def))
        self.workbook_create(self.wb_def, admin=False)
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'workbook-create', params='{0}'.format(self.wb_def))

    def test_wb_isolation(self):
        wb = self.workbook_create(self.wb_def)
        wb_name = self.get_field_value(wb, 'Name')
        wbs = self.mistral_admin('workbook-list')
        self.assertIn(wb_name, [w['Name'] for w in wbs])
        alt_wbs = self.mistral_alt_user('workbook-list')
        self.assertNotIn(wb_name, [w['Name'] for w in alt_wbs])

    def test_get_wb_from_another_tenant(self):
        wb = self.workbook_create(self.wb_def)
        name = self.get_field_value(wb, 'Name')
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'workbook-get', params=name)

    def test_create_public_workbook(self):
        wb = self.workbook_create(self.wb_def, scope='public')
        name = self.get_field_value(wb, 'Name')
        same_wb = self.mistral_alt_user('workbook-get', params=name)
        self.assertEqual(name, self.get_field_value(same_wb, 'Name'))
        self.mistral_alt_user('workflow-get', params='wb.wf1')
        self.mistral_alt_user('action-get', params='wb.ac1')

    def test_delete_wb_from_another_tenant(self):
        wb = self.workbook_create(self.wb_def)
        name = self.get_field_value(wb, 'Name')
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'workbook-delete', params=name)