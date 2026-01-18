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
class TestPacker(TestCaseWithTransport):
    """Tests for the packs repository Packer class."""

    def test_pack_optimizes_pack_order(self):
        builder = self.make_branch_builder('.', format='1.9')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('f', b'f-id', 'file', b'content\n'))], revision_id=b'A')
        builder.build_snapshot([b'A'], [('modify', ('f', b'new-content\n'))], revision_id=b'B')
        builder.build_snapshot([b'B'], [('modify', ('f', b'third-content\n'))], revision_id=b'C')
        builder.build_snapshot([b'C'], [('modify', ('f', b'fourth-content\n'))], revision_id=b'D')
        b = builder.get_branch()
        b.lock_read()
        builder.finish_series()
        self.addCleanup(b.unlock)
        packs = b.repository._pack_collection.packs
        packer = knitpack_repo.KnitPacker(b.repository._pack_collection, packs, 'testing', revision_ids=[b'B', b'C'])
        new_packs = [packs[1], packs[2], packs[0], packs[3]]
        packer.pack()
        self.assertEqual(new_packs, packer.packs)