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
class PullTests(PorcelainTestCase):

    def setUp(self):
        super().setUp()
        handle, fullpath = tempfile.mkstemp(dir=self.repo.path)
        os.close(handle)
        porcelain.add(repo=self.repo.path, paths=fullpath)
        porcelain.commit(repo=self.repo.path, message=b'test', author=b'test <email>', committer=b'test <email>')
        self.target_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.target_path)
        target_repo = porcelain.clone(self.repo.path, target=self.target_path, errstream=BytesIO())
        target_repo.close()
        handle, fullpath = tempfile.mkstemp(dir=self.repo.path)
        os.close(handle)
        porcelain.add(repo=self.repo.path, paths=fullpath)
        porcelain.commit(repo=self.repo.path, message=b'test2', author=b'test2 <email>', committer=b'test2 <email>')
        self.assertIn(b'refs/heads/master', self.repo.refs)
        self.assertIn(b'refs/heads/master', target_repo.refs)

    def test_simple(self):
        outstream = BytesIO()
        errstream = BytesIO()
        porcelain.pull(self.target_path, self.repo.path, b'refs/heads/master', outstream=outstream, errstream=errstream)
        with Repo(self.target_path) as r:
            self.assertEqual(r[b'HEAD'].id, self.repo[b'HEAD'].id)

    def test_diverged(self):
        outstream = BytesIO()
        errstream = BytesIO()
        c3a = porcelain.commit(repo=self.target_path, message=b'test3a', author=b'test2 <email>', committer=b'test2 <email>')
        porcelain.commit(repo=self.repo.path, message=b'test3b', author=b'test2 <email>', committer=b'test2 <email>')
        self.assertRaises(porcelain.DivergedBranches, porcelain.pull, self.target_path, self.repo.path, b'refs/heads/master', outstream=outstream, errstream=errstream)
        with Repo(self.target_path) as r:
            self.assertEqual(r[b'refs/heads/master'].id, c3a)
        self.assertRaises(NotImplementedError, porcelain.pull, self.target_path, self.repo.path, b'refs/heads/master', outstream=outstream, errstream=errstream, fast_forward=False)
        with Repo(self.target_path) as r:
            self.assertEqual(r[b'refs/heads/master'].id, c3a)

    def test_no_refspec(self):
        outstream = BytesIO()
        errstream = BytesIO()
        porcelain.pull(self.target_path, self.repo.path, outstream=outstream, errstream=errstream)
        with Repo(self.target_path) as r:
            self.assertEqual(r[b'HEAD'].id, self.repo[b'HEAD'].id)

    def test_no_remote_location(self):
        outstream = BytesIO()
        errstream = BytesIO()
        porcelain.pull(self.target_path, refspecs=b'refs/heads/master', outstream=outstream, errstream=errstream)
        with Repo(self.target_path) as r:
            self.assertEqual(r[b'HEAD'].id, self.repo[b'HEAD'].id)