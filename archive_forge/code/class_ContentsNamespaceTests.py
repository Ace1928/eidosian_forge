import unittest
import importlib_resources as resources
from . import data01
from . import util
class ContentsNamespaceTests(ContentsTests, unittest.TestCase):
    expected = {'binary.file', 'subdirectory', 'utf-16.file', 'utf-8.file'}

    def setUp(self):
        from . import namespacedata01
        self.data = namespacedata01