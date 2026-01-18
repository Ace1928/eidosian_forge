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
class BranchListTests(PorcelainTestCase):

    def test_standard(self):
        self.assertEqual(set(), set(porcelain.branch_list(self.repo)))

    def test_new_branch(self):
        [c1] = build_commit_graph(self.repo.object_store, [[1]])
        self.repo[b'HEAD'] = c1.id
        porcelain.branch_create(self.repo, b'foo')
        self.assertEqual({b'master', b'foo'}, set(porcelain.branch_list(self.repo)))