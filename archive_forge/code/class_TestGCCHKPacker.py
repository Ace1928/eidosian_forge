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
class TestGCCHKPacker(TestCaseWithTransport):

    def make_abc_branch(self):
        builder = self.make_branch_builder('source')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'A')
        builder.build_snapshot([b'A'], [('add', ('dir', b'dir-id', 'directory', None))], revision_id=b'B')
        builder.build_snapshot([b'B'], [('modify', ('file', b'new content\n'))], revision_id=b'C')
        builder.finish_series()
        return builder.get_branch()

    def make_branch_with_disjoint_inventory_and_revision(self):
        """a repo with separate packs for a revisions Revision and Inventory.

        There will be one pack file that holds the Revision content, and one
        for the Inventory content.

        :return: (repository,
                  pack_name_with_rev_A_Revision,
                  pack_name_with_rev_A_Inventory,
                  pack_name_with_rev_C_content)
        """
        b_source = self.make_abc_branch()
        b_base = b_source.controldir.sprout('base', revision_id=b'A').open_branch()
        b_stacked = b_base.controldir.sprout('stacked', stacked=True).open_branch()
        b_stacked.lock_write()
        self.addCleanup(b_stacked.unlock)
        b_stacked.fetch(b_source, b'B')
        repo_not_stacked = b_stacked.controldir.open_repository()
        repo_not_stacked.lock_write()
        self.addCleanup(repo_not_stacked.unlock)
        self.assertEqual([(b'A',), (b'B',)], sorted(repo_not_stacked.inventories.keys()))
        self.assertEqual([(b'B',)], sorted(repo_not_stacked.revisions.keys()))
        stacked_pack_names = repo_not_stacked._pack_collection.names()
        for name in stacked_pack_names:
            pack = repo_not_stacked._pack_collection.get_pack_by_name(name)
            keys = [n[1] for n in pack.inventory_index.iter_all_entries()]
            if (b'A',) in keys:
                inv_a_pack_name = name
                break
        else:
            self.fail("Could not find pack containing A's inventory")
        repo_not_stacked.fetch(b_source.repository, b'A')
        self.assertEqual([(b'A',), (b'B',)], sorted(repo_not_stacked.revisions.keys()))
        new_pack_names = set(repo_not_stacked._pack_collection.names())
        rev_a_pack_names = new_pack_names.difference(stacked_pack_names)
        self.assertEqual(1, len(rev_a_pack_names))
        rev_a_pack_name = list(rev_a_pack_names)[0]
        repo_not_stacked.fetch(b_source.repository, b'C')
        rev_c_pack_names = set(repo_not_stacked._pack_collection.names())
        rev_c_pack_names = rev_c_pack_names.difference(new_pack_names)
        self.assertEqual(1, len(rev_c_pack_names))
        rev_c_pack_name = list(rev_c_pack_names)[0]
        return (repo_not_stacked, rev_a_pack_name, inv_a_pack_name, rev_c_pack_name)

    def test_pack_with_distant_inventories(self):
        repo, rev_a_pack_name, inv_a_pack_name, rev_c_pack_name = self.make_branch_with_disjoint_inventory_and_revision()
        a_pack = repo._pack_collection.get_pack_by_name(rev_a_pack_name)
        c_pack = repo._pack_collection.get_pack_by_name(rev_c_pack_name)
        packer = groupcompress_repo.GCCHKPacker(repo._pack_collection, [a_pack, c_pack], '.test-pack')
        packer.pack()

    def test_pack_with_missing_inventory(self):
        repo, rev_a_pack_name, inv_a_pack_name, rev_c_pack_name = self.make_branch_with_disjoint_inventory_and_revision()
        inv_a_pack = repo._pack_collection.get_pack_by_name(inv_a_pack_name)
        repo._pack_collection._remove_pack_from_memory(inv_a_pack)
        packer = groupcompress_repo.GCCHKPacker(repo._pack_collection, repo._pack_collection.all_packs(), '.test-pack')
        e = self.assertRaises(ValueError, packer.pack)
        packer.new_pack.abort()
        self.assertContainsRe(str(e), "We are missing inventories for revisions: .*'A'")