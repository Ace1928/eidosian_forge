from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
class TestRepositoryFormatRegistry(TestCase):

    def setUp(self):
        super().setUp()
        self.registry = repository.RepositoryFormatRegistry()

    def test_register_unregister_format(self):
        format = SampleRepositoryFormat()
        self.registry.register(format)
        self.assertEqual(format, self.registry.get(b'Sample .bzr repository format.'))
        self.registry.remove(format)
        self.assertRaises(KeyError, self.registry.get, b'Sample .bzr repository format.')

    def test_get_all(self):
        format = SampleRepositoryFormat()
        self.assertEqual([], self.registry._get_all())
        self.registry.register(format)
        self.assertEqual([format], self.registry._get_all())

    def test_register_extra(self):
        format = SampleExtraRepositoryFormat()
        self.assertEqual([], self.registry._get_all())
        self.registry.register_extra(format)
        self.assertEqual([format], self.registry._get_all())

    def test_register_extra_lazy(self):
        self.assertEqual([], self.registry._get_all())
        self.registry.register_extra_lazy(__name__, 'SampleExtraRepositoryFormat')
        formats = self.registry._get_all()
        self.assertEqual(1, len(formats))
        self.assertIsInstance(formats[0], SampleExtraRepositoryFormat)