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
class PackRefsTests(PorcelainTestCase):

    def test_all(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        self.repo.refs[b'HEAD'] = c3.id
        self.repo.refs[b'refs/heads/master'] = c2.id
        self.repo.refs[b'refs/tags/foo'] = c1.id
        porcelain.pack_refs(self.repo, all=True)
        self.assertEqual(self.repo.refs.get_packed_refs(), {b'refs/heads/master': c2.id, b'refs/tags/foo': c1.id})

    def test_not_all(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        self.repo.refs[b'HEAD'] = c3.id
        self.repo.refs[b'refs/heads/master'] = c2.id
        self.repo.refs[b'refs/tags/foo'] = c1.id
        porcelain.pack_refs(self.repo)
        self.assertEqual(self.repo.refs.get_packed_refs(), {b'refs/tags/foo': c1.id})