from openstackclient.image.v2 import info
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
class TestImportInfo(image_fakes.TestImagev2):
    import_info = image_fakes.create_one_import_info()

    def setUp(self):
        super().setUp()
        self.image_client.get_import_info.return_value = self.import_info
        self.cmd = info.ImportInfo(self.app, None)

    def test_import_info(self):
        arglist = []
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.image_client.get_import_info.assert_called()