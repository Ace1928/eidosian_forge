import bz2
import os
import sys
import tempfile
from io import BytesIO
from ... import diff, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...tests import features, test_commit
from ...tree import InterTree
from .. import bzrdir, inventory, knitrepo
from ..bundle.apply_bundle import install_bundle, merge_bundle
from ..bundle.bundle_data import BundleTree
from ..bundle.serializer import read_bundle, v4, v09, write_bundle
from ..bundle.serializer.v4 import BundleSerializerV4
from ..bundle.serializer.v08 import BundleSerializerV08
from ..bundle.serializer.v09 import BundleSerializerV09
from ..inventorytree import InventoryTree
class V4BundleTester(BundleTester, tests.TestCaseWithTransport):
    format = '4'

    def get_valid_bundle(self, base_rev_id, rev_id, checkout_dir=None):
        """Create a bundle from base_rev_id -> rev_id in built-in branch.
        Make sure that the text generated is valid, and that it
        can be applied against the base, and generate the same information.

        :return: The in-memory bundle
        """
        bundle_txt, rev_ids = self.create_bundle_text(base_rev_id, rev_id)
        bundle = read_bundle(bundle_txt)
        repository = self.b1.repository
        for bundle_rev in bundle.real_revisions:
            branch_rev = repository.get_revision(bundle_rev.revision_id)
            for a in ('inventory_sha1', 'revision_id', 'parent_ids', 'timestamp', 'timezone', 'message', 'committer', 'parent_ids', 'properties'):
                self.assertEqual(getattr(branch_rev, a), getattr(bundle_rev, a))
            self.assertEqual(len(branch_rev.parent_ids), len(bundle_rev.parent_ids))
        self.assertEqual(set(rev_ids), {r.revision_id for r in bundle.real_revisions})
        self.valid_apply_bundle(base_rev_id, bundle, checkout_dir=checkout_dir)
        return bundle

    def get_invalid_bundle(self, base_rev_id, rev_id):
        """Create a bundle from base_rev_id -> rev_id in built-in branch.
        Munge the text so that it's invalid.

        :return: The in-memory bundle
        """
        from ..bundle import serializer
        bundle_txt, rev_ids = self.create_bundle_text(base_rev_id, rev_id)
        new_text = self.get_raw(BytesIO(b''.join(bundle_txt)))
        new_text = new_text.replace(b'<file file_id="exe-1"', b'<file executable="y" file_id="exe-1"')
        new_text = new_text.replace(b'B260', b'B275')
        bundle_txt = BytesIO()
        bundle_txt.write(serializer._get_bundle_header('4'))
        bundle_txt.write(b'\n')
        bundle_txt.write(bz2.compress(new_text))
        bundle_txt.seek(0)
        bundle = read_bundle(bundle_txt)
        self.valid_apply_bundle(base_rev_id, bundle)
        return bundle

    def create_bundle_text(self, base_rev_id, rev_id):
        bundle_txt = BytesIO()
        rev_ids = write_bundle(self.b1.repository, rev_id, base_rev_id, bundle_txt, format=self.format)
        bundle_txt.seek(0)
        self.assertEqual(bundle_txt.readline(), b'# Bazaar revision bundle v%s\n' % self.format.encode('ascii'))
        self.assertEqual(bundle_txt.readline(), b'#\n')
        rev = self.b1.repository.get_revision(rev_id)
        bundle_txt.seek(0)
        return (bundle_txt, rev_ids)

    def get_bundle_tree(self, bundle, revision_id):
        repository = self.make_repository('repo')
        bundle.install_revisions(repository)
        return repository.revision_tree(revision_id)

    def test_creation(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/file', b'contents1\nstatic\n')])
        tree.add('file', ids=b'fileid-2')
        tree.commit('added file', rev_id=b'rev1')
        self.build_tree_contents([('tree/file', b'contents2\nstatic\n')])
        tree.commit('changed file', rev_id=b'rev2')
        s = BytesIO()
        serializer = BundleSerializerV4('1.0')
        with tree.lock_read():
            serializer.write_bundle(tree.branch.repository, b'rev2', b'null:', s)
        s.seek(0)
        tree2 = self.make_branch_and_tree('target')
        target_repo = tree2.branch.repository
        install_bundle(target_repo, serializer.read(s))
        target_repo.lock_read()
        self.addCleanup(target_repo.unlock)
        repo_texts = {i: b''.join(content) for i, content in target_repo.iter_files_bytes([(b'fileid-2', b'rev1', '1'), (b'fileid-2', b'rev2', '2')])}
        self.assertEqual({'1': b'contents1\nstatic\n', '2': b'contents2\nstatic\n'}, repo_texts)
        rtree = target_repo.revision_tree(b'rev2')
        inventory_vf = target_repo.inventories
        self.assertSubset([inventory_vf.get_parent_map([(b'rev2',)])[b'rev2',]], [None, ((b'rev1',),)])
        self.assertEqual('changed file', target_repo.get_revision(b'rev2').message)

    @staticmethod
    def get_raw(bundle_file):
        bundle_file.seek(0)
        line = bundle_file.readline()
        line = bundle_file.readline()
        lines = bundle_file.readlines()
        return bz2.decompress(b''.join(lines))

    def test_copy_signatures(self):
        tree_a = self.make_branch_and_tree('tree_a')
        import breezy.commit as commit
        import breezy.gpg
        oldstrategy = breezy.gpg.GPGStrategy
        branch = tree_a.branch
        repo_a = branch.repository
        tree_a.commit('base', allow_pointless=True, rev_id=b'A')
        self.assertFalse(branch.repository.has_signature_for_revision_id(b'A'))
        try:
            from ..testament import Testament
            breezy.gpg.GPGStrategy = breezy.gpg.LoopbackGPGStrategy
            new_config = test_commit.MustSignConfig()
            commit.Commit(config_stack=new_config).commit(message='base', allow_pointless=True, rev_id=b'B', working_tree=tree_a)

            def sign(text):
                return breezy.gpg.LoopbackGPGStrategy(None).sign(text)
            self.assertTrue(repo_a.has_signature_for_revision_id(b'B'))
        finally:
            breezy.gpg.GPGStrategy = oldstrategy
        tree_b = self.make_branch_and_tree('tree_b')
        repo_b = tree_b.branch.repository
        s = BytesIO()
        serializer = BundleSerializerV4('4')
        with tree_a.lock_read():
            serializer.write_bundle(tree_a.branch.repository, b'B', b'null:', s)
        s.seek(0)
        install_bundle(repo_b, serializer.read(s))
        self.assertTrue(repo_b.has_signature_for_revision_id(b'B'))
        self.assertEqual(repo_b.get_signature_text(b'B'), repo_a.get_signature_text(b'B'))
        s.seek(0)
        install_bundle(repo_b, serializer.read(s))