from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
class TestGetNamedObject(tests.TestCase):
    """Tests for get_named_object."""

    def test_module_only(self):
        import sys
        self.assertIs(sys, get_named_object('sys'))

    def test_dotted_module(self):
        self.assertIs(branch, get_named_object('breezy.branch'))

    def test_module_attr(self):
        self.assertIs(branch.Branch, get_named_object('breezy.branch', 'Branch'))

    def test_dotted_attr(self):
        self.assertIs(branch.Branch.hooks, get_named_object('breezy.branch', 'Branch.hooks'))

    def test_package(self):
        self.assertIs(tests, get_named_object('breezy.tests'))

    def test_package_attr(self):
        self.assertIs(tests.TestCase, get_named_object('breezy.tests', 'TestCase'))

    def test_import_error(self):
        self.assertRaises(ImportError, get_named_object, 'NO_SUCH_MODULE')

    def test_attribute_error(self):
        self.assertRaises(AttributeError, get_named_object, 'sys', 'NO_SUCH_ATTR')