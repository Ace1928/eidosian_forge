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
class CommitTests(PorcelainTestCase):

    def test_custom_author(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        self.repo.refs[b'refs/heads/foo'] = c3.id
        sha = porcelain.commit(self.repo.path, message=b'Some message', author=b'Joe <joe@example.com>', committer=b'Bob <bob@example.com>')
        self.assertIsInstance(sha, bytes)
        self.assertEqual(len(sha), 40)

    def test_unicode(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        self.repo.refs[b'refs/heads/foo'] = c3.id
        sha = porcelain.commit(self.repo.path, message='Some message', author='Joe <joe@example.com>', committer='Bob <bob@example.com>')
        self.assertIsInstance(sha, bytes)
        self.assertEqual(len(sha), 40)

    def test_no_verify(self):
        if os.name != 'posix':
            self.skipTest('shell hook tests requires POSIX shell')
        self.assertTrue(os.path.exists('/bin/sh'))
        hooks_dir = os.path.join(self.repo.controldir(), 'hooks')
        os.makedirs(hooks_dir, exist_ok=True)
        self.addCleanup(shutil.rmtree, hooks_dir)
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        hook_fail = '#!/bin/sh\nexit 1'
        commit_msg = os.path.join(hooks_dir, 'commit-msg')
        with open(commit_msg, 'w') as f:
            f.write(hook_fail)
        os.chmod(commit_msg, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        with self.assertRaises(CommitError):
            porcelain.commit(self.repo.path, message='Some message', author='Joe <joe@example.com>', committer='Bob <bob@example.com>')
        pre_commit = os.path.join(hooks_dir, 'pre-commit')
        with open(pre_commit, 'w') as f:
            f.write(hook_fail)
        os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        with self.assertRaises(CommitError):
            porcelain.commit(self.repo.path, message='Some message', author='Joe <joe@example.com>', committer='Bob <bob@example.com>')
        sha = porcelain.commit(self.repo.path, message='Some message', author='Joe <joe@example.com>', committer='Bob <bob@example.com>', no_verify=True)
        self.assertIsInstance(sha, bytes)
        self.assertEqual(len(sha), 40)

    def test_timezone(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        self.repo.refs[b'refs/heads/foo'] = c3.id
        sha = porcelain.commit(self.repo.path, message='Some message', author='Joe <joe@example.com>', author_timezone=18000, committer='Bob <bob@example.com>', commit_timezone=18000)
        self.assertIsInstance(sha, bytes)
        self.assertEqual(len(sha), 40)
        commit = self.repo.get_object(sha)
        self.assertEqual(commit._author_timezone, 18000)
        self.assertEqual(commit._commit_timezone, 18000)
        self.overrideEnv('GIT_AUTHOR_DATE', '1995-11-20T19:12:08-0501')
        self.overrideEnv('GIT_COMMITTER_DATE', '1995-11-20T19:12:08-0501')
        sha = porcelain.commit(self.repo.path, message='Some message', author='Joe <joe@example.com>', committer='Bob <bob@example.com>')
        self.assertIsInstance(sha, bytes)
        self.assertEqual(len(sha), 40)
        commit = self.repo.get_object(sha)
        self.assertEqual(commit._author_timezone, -18060)
        self.assertEqual(commit._commit_timezone, -18060)
        self.overrideEnv('GIT_AUTHOR_DATE', None)
        self.overrideEnv('GIT_COMMITTER_DATE', None)
        local_timezone = time.localtime().tm_gmtoff
        sha = porcelain.commit(self.repo.path, message='Some message', author='Joe <joe@example.com>', committer='Bob <bob@example.com>')
        self.assertIsInstance(sha, bytes)
        self.assertEqual(len(sha), 40)
        commit = self.repo.get_object(sha)
        self.assertEqual(commit._author_timezone, local_timezone)
        self.assertEqual(commit._commit_timezone, local_timezone)