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
class TagDeleteTests(PorcelainTestCase):

    def test_simple(self):
        [c1] = build_commit_graph(self.repo.object_store, [[1]])
        self.repo[b'HEAD'] = c1.id
        porcelain.tag_create(self.repo, b'foo')
        self.assertIn(b'foo', porcelain.tag_list(self.repo))
        porcelain.tag_delete(self.repo, b'foo')
        self.assertNotIn(b'foo', porcelain.tag_list(self.repo))