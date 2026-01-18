import os
import shutil
import stat
import struct
import sys
import tempfile
from io import BytesIO
from dulwich.tests import TestCase, skipIf
from ..index import (
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..repo import Repo
class GetUnstagedChangesTests(TestCase):

    def test_get_unstaged_changes(self):
        """Unit test for get_unstaged_changes."""
        repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        with Repo.init(repo_dir) as repo:
            foo1_fullpath = os.path.join(repo_dir, 'foo1')
            with open(foo1_fullpath, 'wb') as f:
                f.write(b'origstuff')
            foo2_fullpath = os.path.join(repo_dir, 'foo2')
            with open(foo2_fullpath, 'wb') as f:
                f.write(b'origstuff')
            repo.stage(['foo1', 'foo2'])
            repo.do_commit(b'test status', author=b'author <email>', committer=b'committer <email>')
            with open(foo1_fullpath, 'wb') as f:
                f.write(b'newstuff')
            os.utime(foo1_fullpath, (0, 0))
            changes = get_unstaged_changes(repo.open_index(), repo_dir)
            self.assertEqual(list(changes), [b'foo1'])

    def test_get_unstaged_deleted_changes(self):
        """Unit test for get_unstaged_changes."""
        repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        with Repo.init(repo_dir) as repo:
            foo1_fullpath = os.path.join(repo_dir, 'foo1')
            with open(foo1_fullpath, 'wb') as f:
                f.write(b'origstuff')
            repo.stage(['foo1'])
            repo.do_commit(b'test status', author=b'author <email>', committer=b'committer <email>')
            os.unlink(foo1_fullpath)
            changes = get_unstaged_changes(repo.open_index(), repo_dir)
            self.assertEqual(list(changes), [b'foo1'])

    def test_get_unstaged_changes_removed_replaced_by_directory(self):
        """Unit test for get_unstaged_changes."""
        repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        with Repo.init(repo_dir) as repo:
            foo1_fullpath = os.path.join(repo_dir, 'foo1')
            with open(foo1_fullpath, 'wb') as f:
                f.write(b'origstuff')
            repo.stage(['foo1'])
            repo.do_commit(b'test status', author=b'author <email>', committer=b'committer <email>')
            os.remove(foo1_fullpath)
            os.mkdir(foo1_fullpath)
            changes = get_unstaged_changes(repo.open_index(), repo_dir)
            self.assertEqual(list(changes), [b'foo1'])

    @skipIf(not can_symlink(), 'Requires symlink support')
    def test_get_unstaged_changes_removed_replaced_by_link(self):
        """Unit test for get_unstaged_changes."""
        repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        with Repo.init(repo_dir) as repo:
            foo1_fullpath = os.path.join(repo_dir, 'foo1')
            with open(foo1_fullpath, 'wb') as f:
                f.write(b'origstuff')
            repo.stage(['foo1'])
            repo.do_commit(b'test status', author=b'author <email>', committer=b'committer <email>')
            os.remove(foo1_fullpath)
            os.symlink(os.path.dirname(foo1_fullpath), foo1_fullpath)
            changes = get_unstaged_changes(repo.open_index(), repo_dir)
            self.assertEqual(list(changes), [b'foo1'])