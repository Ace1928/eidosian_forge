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
class TestOptimisingPacker(TestCaseWithTransport):
    """Tests for the OptimisingPacker class."""

    def get_pack_collection(self):
        repo = self.make_repository('.')
        return repo._pack_collection

    def test_open_pack_will_optimise(self):
        packer = knitpack_repo.OptimisingKnitPacker(self.get_pack_collection(), [], '.test')
        new_pack = packer.open_pack()
        self.addCleanup(new_pack.abort)
        self.assertIsInstance(new_pack, pack_repo.NewPack)
        self.assertTrue(new_pack.revision_index._optimize_for_size)
        self.assertTrue(new_pack.inventory_index._optimize_for_size)
        self.assertTrue(new_pack.text_index._optimize_for_size)
        self.assertTrue(new_pack.signature_index._optimize_for_size)