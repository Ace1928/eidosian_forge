import sys
from .. import plugin as _mod_plugin
from .. import symbol_versioning, tests
from . import features
class TestModuleAvailableFeature(tests.TestCase):

    def test_available_module(self):
        feature = features.ModuleAvailableFeature('breezy.tests')
        self.assertEqual('breezy.tests', feature.module_name)
        self.assertEqual('breezy.tests', str(feature))
        self.assertTrue(feature.available())
        self.assertIs(tests, feature.module)

    def test_unavailable_module(self):
        feature = features.ModuleAvailableFeature('breezy.no_such_module_exists')
        self.assertEqual('breezy.no_such_module_exists', str(feature))
        self.assertFalse(feature.available())
        self.assertIs(None, feature.module)