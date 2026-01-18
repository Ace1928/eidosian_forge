from openstackclient.tests.functional.image import base
class InfoTests(base.BaseImageTests):
    """Functional tests for Info commands"""

    def setUp(self):
        super(InfoTests, self).setUp()

    def tearDown(self):
        super().tearDown()

    def test_image_import_info(self):
        output = self.openstack('image import info', parse_output=True)
        self.assertIsNotNone(output['import-methods'])