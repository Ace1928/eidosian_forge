import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
class TestExtensionProperties(utils.TestCase):

    def setUp(self):
        self.ext1 = extension.Extension('name', importlib_metadata.EntryPoint('name', 'module.name:attribute.name [extra]', 'group_name'), mock.Mock(), None)
        self.ext2 = extension.Extension('name', importlib_metadata.EntryPoint('name', 'module:attribute', 'group_name'), mock.Mock(), None)

    def test_module_name(self):
        self.assertEqual('module.name', self.ext1.module_name)
        self.assertEqual('module', self.ext2.module_name)

    def test_attr(self):
        self.assertEqual('attribute.name', self.ext1.attr)
        self.assertEqual('attribute', self.ext2.attr)

    def test_entry_point_target(self):
        self.assertEqual('module.name:attribute.name [extra]', self.ext1.entry_point_target)
        self.assertEqual('module:attribute', self.ext2.entry_point_target)