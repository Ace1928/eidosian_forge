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
class TestKnitPackStreamSource(tests.TestCaseWithMemoryTransport):

    def test_source_to_exact_pack_092(self):
        source = self.make_repository('source', format='pack-0.92')
        target = self.make_repository('target', format='pack-0.92')
        stream_source = source._get_source(target._format)
        self.assertIsInstance(stream_source, knitpack_repo.KnitPackStreamSource)

    def test_source_to_exact_pack_rich_root_pack(self):
        source = self.make_repository('source', format='rich-root-pack')
        target = self.make_repository('target', format='rich-root-pack')
        stream_source = source._get_source(target._format)
        self.assertIsInstance(stream_source, knitpack_repo.KnitPackStreamSource)

    def test_source_to_exact_pack_19(self):
        source = self.make_repository('source', format='1.9')
        target = self.make_repository('target', format='1.9')
        stream_source = source._get_source(target._format)
        self.assertIsInstance(stream_source, knitpack_repo.KnitPackStreamSource)

    def test_source_to_exact_pack_19_rich_root(self):
        source = self.make_repository('source', format='1.9-rich-root')
        target = self.make_repository('target', format='1.9-rich-root')
        stream_source = source._get_source(target._format)
        self.assertIsInstance(stream_source, knitpack_repo.KnitPackStreamSource)

    def test_source_to_remote_exact_pack_19(self):
        trans = self.make_smart_server('target')
        trans.ensure_base()
        source = self.make_repository('source', format='1.9')
        target = self.make_repository('target', format='1.9')
        target = repository.Repository.open(trans.base)
        stream_source = source._get_source(target._format)
        self.assertIsInstance(stream_source, knitpack_repo.KnitPackStreamSource)

    def test_stream_source_to_non_exact(self):
        source = self.make_repository('source', format='pack-0.92')
        target = self.make_repository('target', format='1.9')
        stream = source._get_source(target._format)
        self.assertIs(type(stream), vf_repository.StreamSource)

    def test_stream_source_to_non_exact_rich_root(self):
        source = self.make_repository('source', format='1.9')
        target = self.make_repository('target', format='1.9-rich-root')
        stream = source._get_source(target._format)
        self.assertIs(type(stream), vf_repository.StreamSource)

    def test_source_to_remote_non_exact_pack_19(self):
        trans = self.make_smart_server('target')
        trans.ensure_base()
        source = self.make_repository('source', format='1.9')
        target = self.make_repository('target', format='1.6')
        target = repository.Repository.open(trans.base)
        stream_source = source._get_source(target._format)
        self.assertIs(type(stream_source), vf_repository.StreamSource)

    def test_stream_source_to_knit(self):
        source = self.make_repository('source', format='pack-0.92')
        target = self.make_repository('target', format='dirstate')
        stream = source._get_source(target._format)
        self.assertIs(type(stream), vf_repository.StreamSource)