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
class CloneTests(PorcelainTestCase):

    def test_simple_local(self):
        f1_1 = make_object(Blob, data=b'f1')
        commit_spec = [[1], [2, 1], [3, 1, 2]]
        trees = {1: [(b'f1', f1_1), (b'f2', f1_1)], 2: [(b'f1', f1_1), (b'f2', f1_1)], 3: [(b'f1', f1_1), (b'f2', f1_1)]}
        c1, c2, c3 = build_commit_graph(self.repo.object_store, commit_spec, trees)
        self.repo.refs[b'refs/heads/master'] = c3.id
        self.repo.refs[b'refs/tags/foo'] = c3.id
        target_path = tempfile.mkdtemp()
        errstream = BytesIO()
        self.addCleanup(shutil.rmtree, target_path)
        r = porcelain.clone(self.repo.path, target_path, checkout=False, errstream=errstream)
        self.addCleanup(r.close)
        self.assertEqual(r.path, target_path)
        target_repo = Repo(target_path)
        self.assertEqual(0, len(target_repo.open_index()))
        self.assertEqual(c3.id, target_repo.refs[b'refs/tags/foo'])
        self.assertNotIn(b'f1', os.listdir(target_path))
        self.assertNotIn(b'f2', os.listdir(target_path))
        c = r.get_config()
        encoded_path = self.repo.path
        if not isinstance(encoded_path, bytes):
            encoded_path = encoded_path.encode('utf-8')
        self.assertEqual(encoded_path, c.get((b'remote', b'origin'), b'url'))
        self.assertEqual(b'+refs/heads/*:refs/remotes/origin/*', c.get((b'remote', b'origin'), b'fetch'))

    def test_simple_local_with_checkout(self):
        f1_1 = make_object(Blob, data=b'f1')
        commit_spec = [[1], [2, 1], [3, 1, 2]]
        trees = {1: [(b'f1', f1_1), (b'f2', f1_1)], 2: [(b'f1', f1_1), (b'f2', f1_1)], 3: [(b'f1', f1_1), (b'f2', f1_1)]}
        c1, c2, c3 = build_commit_graph(self.repo.object_store, commit_spec, trees)
        self.repo.refs[b'refs/heads/master'] = c3.id
        target_path = tempfile.mkdtemp()
        errstream = BytesIO()
        self.addCleanup(shutil.rmtree, target_path)
        with porcelain.clone(self.repo.path, target_path, checkout=True, errstream=errstream) as r:
            self.assertEqual(r.path, target_path)
        with Repo(target_path) as r:
            self.assertEqual(r.head(), c3.id)
        self.assertIn('f1', os.listdir(target_path))
        self.assertIn('f2', os.listdir(target_path))

    def test_bare_local_with_checkout(self):
        f1_1 = make_object(Blob, data=b'f1')
        commit_spec = [[1], [2, 1], [3, 1, 2]]
        trees = {1: [(b'f1', f1_1), (b'f2', f1_1)], 2: [(b'f1', f1_1), (b'f2', f1_1)], 3: [(b'f1', f1_1), (b'f2', f1_1)]}
        c1, c2, c3 = build_commit_graph(self.repo.object_store, commit_spec, trees)
        self.repo.refs[b'refs/heads/master'] = c3.id
        target_path = tempfile.mkdtemp()
        errstream = BytesIO()
        self.addCleanup(shutil.rmtree, target_path)
        with porcelain.clone(self.repo.path, target_path, bare=True, errstream=errstream) as r:
            self.assertEqual(r.path, target_path)
        with Repo(target_path) as r:
            r.head()
            self.assertRaises(NoIndexPresent, r.open_index)
        self.assertNotIn(b'f1', os.listdir(target_path))
        self.assertNotIn(b'f2', os.listdir(target_path))

    def test_no_checkout_with_bare(self):
        f1_1 = make_object(Blob, data=b'f1')
        commit_spec = [[1]]
        trees = {1: [(b'f1', f1_1), (b'f2', f1_1)]}
        c1, = build_commit_graph(self.repo.object_store, commit_spec, trees)
        self.repo.refs[b'refs/heads/master'] = c1.id
        self.repo.refs[b'HEAD'] = c1.id
        target_path = tempfile.mkdtemp()
        errstream = BytesIO()
        self.addCleanup(shutil.rmtree, target_path)
        self.assertRaises(porcelain.Error, porcelain.clone, self.repo.path, target_path, checkout=True, bare=True, errstream=errstream)

    def test_no_head_no_checkout(self):
        f1_1 = make_object(Blob, data=b'f1')
        commit_spec = [[1]]
        trees = {1: [(b'f1', f1_1), (b'f2', f1_1)]}
        c1, = build_commit_graph(self.repo.object_store, commit_spec, trees)
        self.repo.refs[b'refs/heads/master'] = c1.id
        target_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, target_path)
        errstream = BytesIO()
        r = porcelain.clone(self.repo.path, target_path, checkout=True, errstream=errstream)
        r.close()

    def test_no_head_no_checkout_outstream_errstream_autofallback(self):
        f1_1 = make_object(Blob, data=b'f1')
        commit_spec = [[1]]
        trees = {1: [(b'f1', f1_1), (b'f2', f1_1)]}
        c1, = build_commit_graph(self.repo.object_store, commit_spec, trees)
        self.repo.refs[b'refs/heads/master'] = c1.id
        target_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, target_path)
        errstream = porcelain.NoneStream()
        r = porcelain.clone(self.repo.path, target_path, checkout=True, errstream=errstream)
        r.close()

    def test_source_broken(self):
        with tempfile.TemporaryDirectory() as parent:
            target_path = os.path.join(parent, 'target')
            self.assertRaises(Exception, porcelain.clone, '/nonexistent/repo', target_path)
            self.assertFalse(os.path.exists(target_path))

    def test_fetch_symref(self):
        f1_1 = make_object(Blob, data=b'f1')
        trees = {1: [(b'f1', f1_1), (b'f2', f1_1)]}
        [c1] = build_commit_graph(self.repo.object_store, [[1]], trees)
        self.repo.refs.set_symbolic_ref(b'HEAD', b'refs/heads/else')
        self.repo.refs[b'refs/heads/else'] = c1.id
        target_path = tempfile.mkdtemp()
        errstream = BytesIO()
        self.addCleanup(shutil.rmtree, target_path)
        r = porcelain.clone(self.repo.path, target_path, checkout=False, errstream=errstream)
        self.addCleanup(r.close)
        self.assertEqual(r.path, target_path)
        target_repo = Repo(target_path)
        self.assertEqual(0, len(target_repo.open_index()))
        self.assertEqual(c1.id, target_repo.refs[b'refs/heads/else'])
        self.assertEqual(c1.id, target_repo.refs[b'HEAD'])
        self.assertEqual({b'HEAD': b'refs/heads/else', b'refs/remotes/origin/HEAD': b'refs/remotes/origin/else'}, target_repo.refs.get_symrefs())

    def test_detached_head(self):
        f1_1 = make_object(Blob, data=b'f1')
        commit_spec = [[1], [2, 1], [3, 1, 2]]
        trees = {1: [(b'f1', f1_1), (b'f2', f1_1)], 2: [(b'f1', f1_1), (b'f2', f1_1)], 3: [(b'f1', f1_1), (b'f2', f1_1)]}
        c1, c2, c3 = build_commit_graph(self.repo.object_store, commit_spec, trees)
        self.repo.refs[b'refs/heads/master'] = c2.id
        self.repo.refs.remove_if_equals(b'HEAD', None)
        self.repo.refs[b'HEAD'] = c3.id
        target_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, target_path)
        errstream = porcelain.NoneStream()
        with porcelain.clone(self.repo.path, target_path, checkout=True, errstream=errstream) as r:
            self.assertEqual(c3.id, r.refs[b'HEAD'])