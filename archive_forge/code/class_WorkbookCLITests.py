import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
class WorkbookCLITests(base_v2.MistralClientTestBase):
    """Test suite checks commands to work with workbooks."""

    @classmethod
    def setUpClass(cls):
        super(WorkbookCLITests, cls).setUpClass()

    def test_workbook_create_delete(self):
        wb = self.mistral_admin('workbook-create', params=self.wb_def)
        wb_name = self.get_field_value(wb, 'Name')
        self.assertTableStruct(wb, ['Field', 'Value'])
        wbs = self.mistral_admin('workbook-list')
        self.assertIn(wb_name, [w['Name'] for w in wbs])
        wbs = self.mistral_admin('workbook-list')
        self.assertIn(wb_name, [w['Name'] for w in wbs])
        self.mistral_admin('workbook-delete', params=wb_name)
        wbs = self.mistral_admin('workbook-list')
        self.assertNotIn(wb_name, [w['Name'] for w in wbs])

    def test_workbook_create_with_tags(self):
        wb = self.workbook_create(self.wb_with_tags_def)
        self.assertIn('tag', self.get_field_value(wb, 'Tags'))

    def test_workbook_update(self):
        wb = self.workbook_create(self.wb_def)
        wb_name = self.get_field_value(wb, 'Name')
        init_update_at = self.get_field_value(wb, 'Updated at')
        tags = self.get_field_value(wb, 'Tags')
        self.assertNotIn('tag', tags)
        wb = self.mistral_admin('workbook-update', params=self.wb_def)
        update_at = self.get_field_value(wb, 'Updated at')
        name = self.get_field_value(wb, 'Name')
        tags = self.get_field_value(wb, 'Tags')
        self.assertEqual(wb_name, name)
        self.assertNotIn('tag', tags)
        self.assertEqual(init_update_at, update_at)
        wb = self.mistral_admin('workbook-update', params=self.wb_with_tags_def)
        self.assertTableStruct(wb, ['Field', 'Value'])
        update_at = self.get_field_value(wb, 'Updated at')
        name = self.get_field_value(wb, 'Name')
        tags = self.get_field_value(wb, 'Tags')
        self.assertEqual(wb_name, name)
        self.assertIn('tag', tags)
        self.assertNotEqual(init_update_at, update_at)

    def test_workbook_get(self):
        created = self.workbook_create(self.wb_with_tags_def)
        wb_name = self.get_field_value(created, 'Name')
        fetched = self.mistral_admin('workbook-get', params=wb_name)
        created_wb_name = self.get_field_value(created, 'Name')
        fetched_wb_name = self.get_field_value(fetched, 'Name')
        self.assertEqual(created_wb_name, fetched_wb_name)
        created_wb_tag = self.get_field_value(created, 'Tags')
        fetched_wb_tag = self.get_field_value(fetched, 'Tags')
        self.assertEqual(created_wb_tag, fetched_wb_tag)

    def test_workbook_get_definition(self):
        wb = self.workbook_create(self.wb_def)
        wb_name = self.get_field_value(wb, 'Name')
        definition = self.mistral_admin('workbook-get-definition', params=wb_name)
        self.assertNotIn('404 Not Found', definition)

    def test_workbook_validate_with_valid_def(self):
        wb = self.mistral_admin('workbook-validate', params=self.wb_def)
        wb_valid = self.get_field_value(wb, 'Valid')
        wb_error = self.get_field_value(wb, 'Error')
        self.assertEqual('True', wb_valid)
        self.assertEqual('None', wb_error)

    def test_workbook_validate_with_invalid_def(self):
        self.create_file('wb.yaml', 'name: wb\n')
        wb = self.mistral_admin('workbook-validate', params='wb.yaml')
        wb_valid = self.get_field_value(wb, 'Valid')
        wb_error = self.get_field_value(wb, 'Error')
        self.assertEqual('False', wb_valid)
        self.assertNotEqual('None', wb_error)