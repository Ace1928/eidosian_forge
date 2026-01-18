from ironicclient.tests.unit import utils
class ImportTest(utils.BaseTestCase):

    def check_exported_symbols(self, exported_symbols):
        self.assertIn('client', exported_symbols)
        self.assertIn('exc', exported_symbols)
        self.assertIn('exceptions', exported_symbols)

    def test_import_objects(self):
        module = __import__(module_str)
        exported_symbols = dir(module)
        self.check_exported_symbols(exported_symbols)

    def test_default_import(self):
        default_imports = __import__(module_str, globals(), locals(), ['*'])
        exported_symbols = dir(default_imports)
        self.check_exported_symbols(exported_symbols)

    def test_import__all__(self):
        module = __import__(module_str)
        self.check_exported_symbols(module.__all__)