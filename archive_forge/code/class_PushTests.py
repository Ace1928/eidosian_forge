import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from io import BytesIO, StringIO
from unittest import skipIf
from dulwich import porcelain
from dulwich.tests import TestCase
from ..diff_tree import tree_changes
from ..errors import CommitError
from ..objects import ZERO_SHA, Blob, Tag, Tree
from ..porcelain import CheckoutError
from ..repo import NoIndexPresent, Repo
from ..server import DictBackend
from ..web import make_server, make_wsgi_chain
from .utils import build_commit_graph, make_commit, make_object
class PushTests(PorcelainTestCase):

    def test_simple(self):
        """Basic test of porcelain push where self.repo is the remote.  First
        clone the remote, commit a file to the clone, then push the changes
        back to the remote.
        """
        outstream = BytesIO()
        errstream = BytesIO()
        porcelain.commit(repo=self.repo.path, message=b'init', author=b'author <email>', committer=b'committer <email>')
        clone_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, clone_path)
        target_repo = porcelain.clone(self.repo.path, target=clone_path, errstream=errstream)
        try:
            self.assertEqual(target_repo[b'HEAD'], self.repo[b'HEAD'])
        finally:
            target_repo.close()
        handle, fullpath = tempfile.mkstemp(dir=clone_path)
        os.close(handle)
        porcelain.add(repo=clone_path, paths=[fullpath])
        porcelain.commit(repo=clone_path, message=b'push', author=b'author <email>', committer=b'committer <email>')
        refs_path = b'refs/heads/foo'
        new_id = self.repo[b'HEAD'].id
        self.assertNotEqual(new_id, ZERO_SHA)
        self.repo.refs[refs_path] = new_id
        porcelain.push(clone_path, 'origin', b'HEAD:' + refs_path, outstream=outstream, errstream=errstream)
        self.assertEqual(target_repo.refs[b'refs/remotes/origin/foo'], target_repo.refs[b'HEAD'])
        with Repo(clone_path) as r_clone:
            self.assertEqual({b'HEAD': new_id, b'refs/heads/foo': r_clone[b'HEAD'].id, b'refs/heads/master': new_id}, self.repo.get_refs())
            self.assertEqual(r_clone[b'HEAD'].id, self.repo[refs_path].id)
            change = next(iter(tree_changes(self.repo, self.repo[b'HEAD'].tree, self.repo[b'refs/heads/foo'].tree)))
            self.assertEqual(os.path.basename(fullpath), change.new.path.decode('ascii'))

    def test_local_missing(self):
        """Pushing a new branch."""
        outstream = BytesIO()
        errstream = BytesIO()
        clone_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, clone_path)
        target_repo = porcelain.init(clone_path)
        target_repo.close()
        self.assertRaises(porcelain.Error, porcelain.push, self.repo, clone_path, b'HEAD:refs/heads/master', outstream=outstream, errstream=errstream)

    def test_new(self):
        """Pushing a new branch."""
        outstream = BytesIO()
        errstream = BytesIO()
        clone_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, clone_path)
        target_repo = porcelain.init(clone_path)
        target_repo.close()
        handle, fullpath = tempfile.mkstemp(dir=clone_path)
        os.close(handle)
        porcelain.add(repo=clone_path, paths=[fullpath])
        new_id = porcelain.commit(repo=self.repo, message=b'push', author=b'author <email>', committer=b'committer <email>')
        porcelain.push(self.repo, clone_path, b'HEAD:refs/heads/master', outstream=outstream, errstream=errstream)
        with Repo(clone_path) as r_clone:
            self.assertEqual({b'HEAD': new_id, b'refs/heads/master': new_id}, r_clone.get_refs())

    def test_delete(self):
        """Basic test of porcelain push, removing a branch."""
        outstream = BytesIO()
        errstream = BytesIO()
        porcelain.commit(repo=self.repo.path, message=b'init', author=b'author <email>', committer=b'committer <email>')
        clone_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, clone_path)
        target_repo = porcelain.clone(self.repo.path, target=clone_path, errstream=errstream)
        target_repo.close()
        refs_path = b'refs/heads/foo'
        new_id = self.repo[b'HEAD'].id
        self.assertNotEqual(new_id, ZERO_SHA)
        self.repo.refs[refs_path] = new_id
        porcelain.push(clone_path, self.repo.path, b':' + refs_path, outstream=outstream, errstream=errstream)
        self.assertEqual({b'HEAD': new_id, b'refs/heads/master': new_id}, self.repo.get_refs())

    def test_diverged(self):
        outstream = BytesIO()
        errstream = BytesIO()
        porcelain.commit(repo=self.repo.path, message=b'init', author=b'author <email>', committer=b'committer <email>')
        clone_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, clone_path)
        target_repo = porcelain.clone(self.repo.path, target=clone_path, errstream=errstream)
        target_repo.close()
        remote_id = porcelain.commit(repo=self.repo.path, message=b'remote change', author=b'author <email>', committer=b'committer <email>')
        local_id = porcelain.commit(repo=clone_path, message=b'local change', author=b'author <email>', committer=b'committer <email>')
        outstream = BytesIO()
        errstream = BytesIO()
        self.assertRaises(porcelain.DivergedBranches, porcelain.push, clone_path, self.repo.path, b'refs/heads/master', outstream=outstream, errstream=errstream)
        self.assertEqual({b'HEAD': remote_id, b'refs/heads/master': remote_id}, self.repo.get_refs())
        self.assertEqual(b'', outstream.getvalue())
        self.assertEqual(b'', errstream.getvalue())
        outstream = BytesIO()
        errstream = BytesIO()
        porcelain.push(clone_path, self.repo.path, b'refs/heads/master', outstream=outstream, errstream=errstream, force=True)
        self.assertEqual({b'HEAD': local_id, b'refs/heads/master': local_id}, self.repo.get_refs())
        self.assertEqual(b'', outstream.getvalue())
        self.assertTrue(re.match(b'Push to .* successful.\n', errstream.getvalue()))