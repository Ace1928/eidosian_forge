import uuid
from osc_placement.tests.functional import base
class TestResourceClass(base.BaseTestCase):
    VERSION = '1.2'

    def test_list(self):
        rcs = self.resource_class_list()
        names = [rc['name'] for rc in rcs]
        self.assertIn('VCPU', names)
        self.assertIn('MEMORY_MB', names)
        self.assertIn('DISK_GB', names)

    def test_fail_create_if_incorrect_class(self):
        self.assertCommandFailed('JSON does not validate', self.resource_class_create, 'fake_class')
        self.assertCommandFailed('JSON does not validate', self.resource_class_create, 'CUSTOM_lower')
        self.assertCommandFailed('JSON does not validate', self.resource_class_create, 'CUSTOM_GPU.INTEL')

    def test_create(self):
        self.resource_class_create(CUSTOM_RC)
        rcs = self.resource_class_list()
        names = [rc['name'] for rc in rcs]
        self.assertIn(CUSTOM_RC, names)
        self.resource_class_delete(CUSTOM_RC)

    def test_fail_show_if_unknown_class(self):
        self.assertCommandFailed('No such resource class', self.resource_class_show, 'UNKNOWN')

    def test_show(self):
        rc = self.resource_class_show('VCPU')
        self.assertEqual('VCPU', rc['name'])

    def test_fail_delete_unknown_class(self):
        self.assertCommandFailed('No such resource class', self.resource_class_delete, 'UNKNOWN')

    def test_fail_delete_standard_class(self):
        self.assertCommandFailed('Cannot delete standard resource class', self.resource_class_delete, 'VCPU')