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
class PathToTreeTests(PorcelainTestCase):

    def setUp(self):
        super().setUp()
        self.fp = os.path.join(self.test_dir, 'bar')
        with open(self.fp, 'w') as f:
            f.write('something')
        oldcwd = os.getcwd()
        self.addCleanup(os.chdir, oldcwd)
        os.chdir(self.test_dir)

    def test_path_to_tree_path_base(self):
        self.assertEqual(b'bar', porcelain.path_to_tree_path(self.test_dir, self.fp))
        self.assertEqual(b'bar', porcelain.path_to_tree_path('.', './bar'))
        self.assertEqual(b'bar', porcelain.path_to_tree_path('.', 'bar'))
        cwd = os.getcwd()
        self.assertEqual(b'bar', porcelain.path_to_tree_path('.', os.path.join(cwd, 'bar')))
        self.assertEqual(b'bar', porcelain.path_to_tree_path(cwd, 'bar'))

    def test_path_to_tree_path_syntax(self):
        self.assertEqual(b'bar', porcelain.path_to_tree_path('.', './bar'))

    def test_path_to_tree_path_error(self):
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as od:
                porcelain.path_to_tree_path(od, self.fp)

    def test_path_to_tree_path_rel(self):
        cwd = os.getcwd()
        os.mkdir(os.path.join(self.repo.path, 'foo'))
        os.mkdir(os.path.join(self.repo.path, 'foo/bar'))
        try:
            os.chdir(os.path.join(self.repo.path, 'foo/bar'))
            with open('baz', 'w') as f:
                f.write('contents')
            self.assertEqual(b'bar/baz', porcelain.path_to_tree_path('..', 'baz'))
            self.assertEqual(b'bar/baz', porcelain.path_to_tree_path(os.path.join(os.getcwd(), '..'), os.path.join(os.getcwd(), 'baz')))
            self.assertEqual(b'bar/baz', porcelain.path_to_tree_path('..', os.path.join(os.getcwd(), 'baz')))
            self.assertEqual(b'bar/baz', porcelain.path_to_tree_path(os.path.join(os.getcwd(), '..'), 'baz'))
        finally:
            os.chdir(cwd)