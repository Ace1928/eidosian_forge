import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
class FileIdInvolvedWGhosts(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def create_branch_with_ghost_text(self):
        builder = self.make_branch_builder('ghost')
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('a', b'a-file-id', 'file', b'some content\n'))], revision_id=b'A-id')
        b = builder.get_branch()
        old_rt = b.repository.revision_tree(b'A-id')
        new_inv = inventory.mutable_inventory_from_tree(old_rt)
        new_inv.revision_id = b'B-id'
        new_inv.get_entry(b'a-file-id').revision = b'ghost-id'
        new_rev = _mod_revision.Revision(b'B-id', timestamp=time.time(), timezone=0, message='Committing against a ghost', committer='Joe Foo <joe@foo.com>', properties={}, parent_ids=(b'A-id', b'ghost-id'))
        b.lock_write()
        self.addCleanup(b.unlock)
        b.repository.start_write_group()
        b.repository.add_revision(b'B-id', new_rev, new_inv)
        self.disable_commit_write_group_paranoia(b.repository)
        b.repository.commit_write_group()
        return b

    def disable_commit_write_group_paranoia(self, repo):
        if isinstance(repo, remote.RemoteRepository):
            repo.abort_write_group()
            raise tests.TestSkipped('repository format does not support storing revisions with missing texts.')
        pack_coll = getattr(repo, '_pack_collection', None)
        if pack_coll is not None:
            pack_coll._check_new_inventories = lambda: []

    def test_file_ids_include_ghosts(self):
        b = self.create_branch_with_ghost_text()
        repo = b.repository
        self.assertEqual({b'a-file-id': {b'ghost-id'}}, repo.fileids_altered_by_revision_ids([b'B-id']))

    def test_file_ids_uses_fallbacks(self):
        builder = self.make_branch_builder('source', format=self.bzrdir_format)
        repo = builder.get_branch().repository
        if not repo._format.supports_external_lookups:
            raise tests.TestNotApplicable('format does not support stacking')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'contents\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [('modify', ('file', b'new-content\n'))], revision_id=b'B-id')
        builder.build_snapshot([b'B-id'], [('modify', ('file', b'yet more content\n'))], revision_id=b'C-id')
        builder.finish_series()
        source_b = builder.get_branch()
        source_b.lock_read()
        self.addCleanup(source_b.unlock)
        base = self.make_branch('base')
        base.pull(source_b, stop_revision=b'B-id')
        stacked = self.make_branch('stacked')
        stacked.set_stacked_on_url('../base')
        stacked.pull(source_b, stop_revision=b'C-id')
        stacked.lock_read()
        self.addCleanup(stacked.unlock)
        repo = stacked.repository
        keys = {b'file-id': {b'A-id'}}
        if stacked.repository.supports_rich_root():
            keys[b'root-id'] = {b'A-id'}
        self.assertEqual(keys, repo.fileids_altered_by_revision_ids([b'A-id']))