from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
class TestOldDirectoryLookup(TestCase):
    """Test compatibility with older implementations of Directory
    that don't support the purpose argument."""

    def setUp(self):
        super().setUp()
        self.registry = DirectoryServiceRegistry()
        self.registry.register('old:', OldService, 'Map foo URLs to http urls')

    def test_dereference(self):
        self.assertEqual(OldService.base + 'bar', self.registry.dereference('old:bar'))
        self.assertEqual(OldService.base + 'bar', self.registry.dereference('old:bar', purpose='write'))
        self.assertEqual('baz:qux', self.registry.dereference('baz:qux'))
        self.assertEqual('baz:qux', self.registry.dereference('baz:qux', purpose='write'))