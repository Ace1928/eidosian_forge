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
class LsTreeTests(PorcelainTestCase):

    def test_empty(self):
        porcelain.commit(repo=self.repo.path, message=b'test status', author=b'author <email>', committer=b'committer <email>')
        f = StringIO()
        porcelain.ls_tree(self.repo, b'HEAD', outstream=f)
        self.assertEqual(f.getvalue(), '')

    def test_simple(self):
        fullpath = os.path.join(self.repo.path, 'foo')
        with open(fullpath, 'w') as f:
            f.write('origstuff')
        porcelain.add(repo=self.repo.path, paths=[fullpath])
        porcelain.commit(repo=self.repo.path, message=b'test status', author=b'author <email>', committer=b'committer <email>')
        f = StringIO()
        porcelain.ls_tree(self.repo, b'HEAD', outstream=f)
        self.assertEqual(f.getvalue(), '100644 blob 8b82634d7eae019850bb883f06abf428c58bc9aa\tfoo\n')

    def test_recursive(self):
        dirpath = os.path.join(self.repo.path, 'adir')
        filepath = os.path.join(dirpath, 'afile')
        os.mkdir(dirpath)
        with open(filepath, 'w') as f:
            f.write('origstuff')
        porcelain.add(repo=self.repo.path, paths=[filepath])
        porcelain.commit(repo=self.repo.path, message=b'test status', author=b'author <email>', committer=b'committer <email>')
        f = StringIO()
        porcelain.ls_tree(self.repo, b'HEAD', outstream=f)
        self.assertEqual(f.getvalue(), '40000 tree b145cc69a5e17693e24d8a7be0016ed8075de66d\tadir\n')
        f = StringIO()
        porcelain.ls_tree(self.repo, b'HEAD', outstream=f, recursive=True)
        self.assertEqual(f.getvalue(), '40000 tree b145cc69a5e17693e24d8a7be0016ed8075de66d\tadir\n100644 blob 8b82634d7eae019850bb883f06abf428c58bc9aa\tadir/afile\n')