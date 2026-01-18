from breezy import errors, osutils, repository
from breezy.bzr import btree_index
from breezy.bzr.remote import RemoteRepository
from breezy.bzr.tests.per_repository_chk import TestCaseWithRepositoryCHK
from breezy.bzr.versionedfile import VersionedFiles
from breezy.tests import TestNotApplicable
class TestCommitWriteGroupIntegrityCheck(TestCaseWithRepositoryCHK):
    """Tests that commit_write_group prevents various kinds of invalid data
    from being committed to a CHK repository.
    """

    def reopen_repo_and_resume_write_group(self, repo):
        resume_tokens = repo.suspend_write_group()
        repo.unlock()
        reopened_repo = repo.controldir.open_repository()
        reopened_repo.lock_write()
        self.addCleanup(reopened_repo.unlock)
        reopened_repo.resume_write_group(resume_tokens)
        return reopened_repo

    def test_missing_chk_root_for_inventory(self):
        """commit_write_group fails with BzrCheckError when the chk root record
        for a new inventory is missing.
        """
        repo = self.make_repository('damaged-repo')
        builder = self.make_branch_builder('simple-branch')
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'A-id')
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        repo.lock_write()
        repo.start_write_group()
        text_keys = [(b'file-id', b'A-id'), (b'root-id', b'A-id')]
        src_repo = b.repository
        repo.texts.insert_record_stream(src_repo.texts.get_record_stream(text_keys, 'unordered', True))
        repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([(b'A-id',)], 'unordered', True))
        repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([(b'A-id',)], 'unordered', True))
        repo.add_fallback_repository(b.repository)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        reopened_repo = self.reopen_repo_and_resume_write_group(repo)
        self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
        reopened_repo.abort_write_group()

    def test_missing_chk_root_for_unchanged_inventory(self):
        """commit_write_group fails with BzrCheckError when the chk root record
        for a new inventory is missing, even if the parent inventory is present
        and has identical content (i.e. the same chk root).

        A stacked repository containing only a revision with an identical
        inventory to its parent will still have the chk root records for those
        inventories.

        (In principle the chk records are unnecessary in this case, but in
        practice bzr 2.0rc1 (at least) expects to find them.)
        """
        repo = self.make_repository('damaged-repo')
        builder = self.make_branch_builder('simple-branch')
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'A-id')
        builder.build_snapshot(None, [], revision_id=b'B-id')
        builder.build_snapshot(None, [], revision_id=b'C-id')
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        inv_b = b.repository.get_inventory(b'B-id')
        inv_c = b.repository.get_inventory(b'C-id')
        if not isinstance(repo, RemoteRepository):
            self.assertEqual(inv_b.id_to_entry.key(), inv_c.id_to_entry.key())
        repo.lock_write()
        repo.start_write_group()
        src_repo = b.repository
        repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([(b'B-id',), (b'C-id',)], 'unordered', True))
        repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([(b'C-id',)], 'unordered', True))
        repo.add_fallback_repository(b.repository)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        reopened_repo = self.reopen_repo_and_resume_write_group(repo)
        self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
        reopened_repo.abort_write_group()

    def test_missing_chk_leaf_for_inventory(self):
        """commit_write_group fails with BzrCheckError when the chk root record
        for a parent inventory of a new revision is missing.
        """
        repo = self.make_repository('damaged-repo')
        if isinstance(repo, RemoteRepository):
            raise TestNotApplicable('Unable to obtain CHKInventory from remote repo')
        b = self.make_branch_with_multiple_chk_nodes()
        src_repo = b.repository
        src_repo.lock_read()
        self.addCleanup(src_repo.unlock)
        inv_b = src_repo.get_inventory(b'B-id')
        inv_c = src_repo.get_inventory(b'C-id')
        chk_root_keys_only = [inv_b.id_to_entry.key(), inv_b.parent_id_basename_to_file_id.key(), inv_c.id_to_entry.key(), inv_c.parent_id_basename_to_file_id.key()]
        all_chks = src_repo.chk_bytes.keys()
        for key_to_drop in all_chks.difference(chk_root_keys_only):
            all_chks.discard(key_to_drop)
        repo.lock_write()
        repo.start_write_group()
        repo.chk_bytes.insert_record_stream(src_repo.chk_bytes.get_record_stream(all_chks, 'unordered', True))
        repo.texts.insert_record_stream(src_repo.texts.get_record_stream(src_repo.texts.keys(), 'unordered', True))
        repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([(b'B-id',), (b'C-id',)], 'unordered', True))
        repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([(b'C-id',)], 'unordered', True))
        repo.add_fallback_repository(b.repository)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        reopened_repo = self.reopen_repo_and_resume_write_group(repo)
        self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
        reopened_repo.abort_write_group()

    def test_missing_chk_root_for_parent_inventory(self):
        """commit_write_group fails with BzrCheckError when the chk root record
        for a parent inventory of a new revision is missing.
        """
        repo = self.make_repository('damaged-repo')
        if isinstance(repo, RemoteRepository):
            raise TestNotApplicable('Unable to obtain CHKInventory from remote repo')
        b = self.make_branch_with_multiple_chk_nodes()
        b.lock_read()
        self.addCleanup(b.unlock)
        inv_c = b.repository.get_inventory(b'C-id')
        chk_keys_for_c_only = [inv_c.id_to_entry.key(), inv_c.parent_id_basename_to_file_id.key()]
        repo.lock_write()
        repo.start_write_group()
        src_repo = b.repository
        repo.chk_bytes.insert_record_stream(src_repo.chk_bytes.get_record_stream(chk_keys_for_c_only, 'unordered', True))
        repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([(b'B-id',), (b'C-id',)], 'unordered', True))
        repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([(b'C-id',)], 'unordered', True))
        repo.add_fallback_repository(b.repository)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        reopened_repo = self.reopen_repo_and_resume_write_group(repo)
        self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
        reopened_repo.abort_write_group()

    def make_branch_with_multiple_chk_nodes(self):
        builder = self.make_branch_builder('simple-branch')
        file_adds = []
        file_modifies = []
        for char in 'abc':
            name = char * 10000
            file_adds.append(('add', ('file-' + name, ('file-%s-id' % name).encode(), 'file', ('content %s\n' % name).encode())))
            file_modifies.append(('modify', ('file-' + name, ('new content %s\n' % name).encode())))
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None))] + file_adds, revision_id=b'A-id')
        builder.build_snapshot(None, [], revision_id=b'B-id')
        builder.build_snapshot(None, file_modifies, revision_id=b'C-id')
        return builder.get_branch()

    def test_missing_text_record(self):
        """commit_write_group fails with BzrCheckError when a text is missing.
        """
        repo = self.make_repository('damaged-repo')
        b = self.make_branch_with_multiple_chk_nodes()
        src_repo = b.repository
        src_repo.lock_read()
        self.addCleanup(src_repo.unlock)
        all_texts = src_repo.texts.keys()
        all_texts.remove((b'file-%s-id' % (b'c' * 10000,), b'C-id'))
        repo.lock_write()
        repo.start_write_group()
        repo.chk_bytes.insert_record_stream(src_repo.chk_bytes.get_record_stream(src_repo.chk_bytes.keys(), 'unordered', True))
        repo.texts.insert_record_stream(src_repo.texts.get_record_stream(all_texts, 'unordered', True))
        repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([(b'B-id',), (b'C-id',)], 'unordered', True))
        repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([(b'C-id',)], 'unordered', True))
        repo.add_fallback_repository(b.repository)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        reopened_repo = self.reopen_repo_and_resume_write_group(repo)
        self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
        reopened_repo.abort_write_group()