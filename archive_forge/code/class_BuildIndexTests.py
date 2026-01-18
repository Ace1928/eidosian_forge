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
class BuildIndexTests(TestCase):

    def assertReasonableIndexEntry(self, index_entry, mode, filesize, sha):
        self.assertEqual(index_entry.mode, mode)
        self.assertEqual(index_entry.size, filesize)
        self.assertEqual(index_entry.sha, sha)

    def assertFileContents(self, path, contents, symlink=False):
        if symlink:
            self.assertEqual(os.readlink(path), contents)
        else:
            with open(path, 'rb') as f:
                self.assertEqual(f.read(), contents)

    def test_empty(self):
        repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        with Repo.init(repo_dir) as repo:
            tree = Tree()
            repo.object_store.add_object(tree)
            build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
            index = repo.open_index()
            self.assertEqual(len(index), 0)
            self.assertEqual(['.git'], os.listdir(repo.path))

    def test_git_dir(self):
        repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        with Repo.init(repo_dir) as repo:
            filea = Blob.from_string(b'file a')
            filee = Blob.from_string(b'd')
            tree = Tree()
            tree[b'.git/a'] = (stat.S_IFREG | 420, filea.id)
            tree[b'c/e'] = (stat.S_IFREG | 420, filee.id)
            repo.object_store.add_objects([(o, None) for o in [filea, filee, tree]])
            build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
            index = repo.open_index()
            self.assertEqual(len(index), 1)
            apath = os.path.join(repo.path, '.git', 'a')
            self.assertFalse(os.path.exists(apath))
            epath = os.path.join(repo.path, 'c', 'e')
            self.assertTrue(os.path.exists(epath))
            self.assertReasonableIndexEntry(index[b'c/e'], stat.S_IFREG | 420, 1, filee.id)
            self.assertFileContents(epath, b'd')

    def test_nonempty(self):
        repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        with Repo.init(repo_dir) as repo:
            filea = Blob.from_string(b'file a')
            fileb = Blob.from_string(b'file b')
            filed = Blob.from_string(b'file d')
            tree = Tree()
            tree[b'a'] = (stat.S_IFREG | 420, filea.id)
            tree[b'b'] = (stat.S_IFREG | 420, fileb.id)
            tree[b'c/d'] = (stat.S_IFREG | 420, filed.id)
            repo.object_store.add_objects([(o, None) for o in [filea, fileb, filed, tree]])
            build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
            index = repo.open_index()
            self.assertEqual(len(index), 3)
            apath = os.path.join(repo.path, 'a')
            self.assertTrue(os.path.exists(apath))
            self.assertReasonableIndexEntry(index[b'a'], stat.S_IFREG | 420, 6, filea.id)
            self.assertFileContents(apath, b'file a')
            bpath = os.path.join(repo.path, 'b')
            self.assertTrue(os.path.exists(bpath))
            self.assertReasonableIndexEntry(index[b'b'], stat.S_IFREG | 420, 6, fileb.id)
            self.assertFileContents(bpath, b'file b')
            dpath = os.path.join(repo.path, 'c', 'd')
            self.assertTrue(os.path.exists(dpath))
            self.assertReasonableIndexEntry(index[b'c/d'], stat.S_IFREG | 420, 6, filed.id)
            self.assertFileContents(dpath, b'file d')
            self.assertEqual(['.git', 'a', 'b', 'c'], sorted(os.listdir(repo.path)))
            self.assertEqual(['d'], sorted(os.listdir(os.path.join(repo.path, 'c'))))

    @skipIf(not getattr(os, 'sync', None), 'Requires sync support')
    def test_norewrite(self):
        repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        with Repo.init(repo_dir) as repo:
            filea = Blob.from_string(b'file a')
            filea_path = os.path.join(repo_dir, 'a')
            tree = Tree()
            tree[b'a'] = (stat.S_IFREG | 420, filea.id)
            repo.object_store.add_objects([(o, None) for o in [filea, tree]])
            build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
            os.sync()
            mtime = os.stat(filea_path).st_mtime
            build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
            os.sync()
            self.assertEqual(mtime, os.stat(filea_path).st_mtime)
            with open(filea_path, 'wb') as fh:
                fh.write(b'test a')
            os.sync()
            mtime = os.stat(filea_path).st_mtime
            build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
            os.sync()
            with open(filea_path, 'rb') as fh:
                self.assertEqual(b'file a', fh.read())

    @skipIf(not can_symlink(), 'Requires symlink support')
    def test_symlink(self):
        repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        with Repo.init(repo_dir) as repo:
            filed = Blob.from_string(b'file d')
            filee = Blob.from_string(b'd')
            tree = Tree()
            tree[b'c/d'] = (stat.S_IFREG | 420, filed.id)
            tree[b'c/e'] = (stat.S_IFLNK, filee.id)
            repo.object_store.add_objects([(o, None) for o in [filed, filee, tree]])
            build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
            index = repo.open_index()
            epath = os.path.join(repo.path, 'c', 'e')
            self.assertTrue(os.path.exists(epath))
            self.assertReasonableIndexEntry(index[b'c/e'], stat.S_IFLNK, 0 if sys.platform == 'win32' else 1, filee.id)
            self.assertFileContents(epath, 'd', symlink=True)

    def test_no_decode_encode(self):
        repo_dir = tempfile.mkdtemp()
        repo_dir_bytes = os.fsencode(repo_dir)
        self.addCleanup(shutil.rmtree, repo_dir)
        with Repo.init(repo_dir) as repo:
            file = Blob.from_string(b'foo')
            tree = Tree()
            latin1_name = 'À'.encode('latin1')
            latin1_path = os.path.join(repo_dir_bytes, latin1_name)
            utf8_name = 'À'.encode()
            utf8_path = os.path.join(repo_dir_bytes, utf8_name)
            tree[latin1_name] = (stat.S_IFREG | 420, file.id)
            tree[utf8_name] = (stat.S_IFREG | 420, file.id)
            repo.object_store.add_objects([(o, None) for o in [file, tree]])
            try:
                build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
            except OSError as e:
                if e.errno == 92 and sys.platform == 'darwin':
                    self.skipTest('can not write filename %r' % e.filename)
                else:
                    raise
            except UnicodeDecodeError:
                self.skipTest('can not implicitly convert as utf8')
            index = repo.open_index()
            self.assertIn(latin1_name, index)
            self.assertIn(utf8_name, index)
            self.assertTrue(os.path.exists(latin1_path))
            self.assertTrue(os.path.exists(utf8_path))

    def test_git_submodule(self):
        repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        with Repo.init(repo_dir) as repo:
            filea = Blob.from_string(b'file alalala')
            subtree = Tree()
            subtree[b'a'] = (stat.S_IFREG | 420, filea.id)
            c = Commit()
            c.tree = subtree.id
            c.committer = c.author = b'Somebody <somebody@example.com>'
            c.commit_time = c.author_time = 42342
            c.commit_timezone = c.author_timezone = 0
            c.parents = []
            c.message = b'Subcommit'
            tree = Tree()
            tree[b'c'] = (S_IFGITLINK, c.id)
            repo.object_store.add_objects([(o, None) for o in [tree]])
            build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
            index = repo.open_index()
            self.assertEqual(len(index), 1)
            apath = os.path.join(repo.path, 'c/a')
            self.assertFalse(os.path.exists(apath))
            cpath = os.path.join(repo.path, 'c')
            self.assertTrue(os.path.isdir(cpath))
            self.assertEqual(index[b'c'].mode, S_IFGITLINK)
            self.assertEqual(index[b'c'].sha, c.id)

    def test_git_submodule_exists(self):
        repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        with Repo.init(repo_dir) as repo:
            filea = Blob.from_string(b'file alalala')
            subtree = Tree()
            subtree[b'a'] = (stat.S_IFREG | 420, filea.id)
            c = Commit()
            c.tree = subtree.id
            c.committer = c.author = b'Somebody <somebody@example.com>'
            c.commit_time = c.author_time = 42342
            c.commit_timezone = c.author_timezone = 0
            c.parents = []
            c.message = b'Subcommit'
            tree = Tree()
            tree[b'c'] = (S_IFGITLINK, c.id)
            os.mkdir(os.path.join(repo_dir, 'c'))
            repo.object_store.add_objects([(o, None) for o in [tree]])
            build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
            index = repo.open_index()
            self.assertEqual(len(index), 1)
            apath = os.path.join(repo.path, 'c/a')
            self.assertFalse(os.path.exists(apath))
            cpath = os.path.join(repo.path, 'c')
            self.assertTrue(os.path.isdir(cpath))
            self.assertEqual(index[b'c'].mode, S_IFGITLINK)
            self.assertEqual(index[b'c'].sha, c.id)