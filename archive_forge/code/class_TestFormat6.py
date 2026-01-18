import sys
from io import BytesIO
from stat import S_ISDIR
from ...bzr.bzrdir import BzrDirMetaFormat1
from ...bzr.serializer import format_registry as serializer_format_registry
from ...errors import IllegalPath
from ...repository import InterRepository, Repository
from ...tests import TestCase, TestCaseWithTransport
from ...transport import NoSuchFile
from . import xml4
from .bzrdir import BzrDirFormat6
from .repository import (InterWeaveRepo, RepositoryFormat4, RepositoryFormat5,
class TestFormat6(TestCaseWithTransport):

    def test_attribute__fetch_order(self):
        """Weaves need topological data insertion."""
        control = BzrDirFormat6().initialize(self.get_url())
        repo = RepositoryFormat6().initialize(control)
        self.assertEqual('topological', repo._format._fetch_order)

    def test_attribute__fetch_uses_deltas(self):
        """Weaves do not reuse deltas."""
        control = BzrDirFormat6().initialize(self.get_url())
        repo = RepositoryFormat6().initialize(control)
        self.assertEqual(False, repo._format._fetch_uses_deltas)

    def test_attribute__fetch_reconcile(self):
        """Weave repositories need a reconcile after fetch."""
        control = BzrDirFormat6().initialize(self.get_url())
        repo = RepositoryFormat6().initialize(control)
        self.assertEqual(True, repo._format._fetch_reconcile)

    def test_no_ancestry_weave(self):
        control = BzrDirFormat6().initialize(self.get_url())
        repo = RepositoryFormat6().initialize(control)
        self.assertRaises(NoSuchFile, control.transport.get, 'ancestry.weave')

    def test_supports_external_lookups(self):
        control = BzrDirFormat6().initialize(self.get_url())
        repo = RepositoryFormat6().initialize(control)
        self.assertFalse(repo._format.supports_external_lookups)